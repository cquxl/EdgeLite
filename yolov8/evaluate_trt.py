import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
from numpy import ndarray
import torch
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import sys
import re
from tqdm import tqdm 
import time
import cv2
import numpy as np
from utils.config import COLORS, KPS_COLORS, LIMB_COLORS, SKELETON

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolov8_pose_infer import TRTModule
from torchvision.ops import nms, batched_nms
from ultralytics.utils import ops  # ultralytics 内部 CUDA NMS

BASE_DIR = Path(__file__).resolve().parent

SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff',
           '.webp', '.pfm')

# coco-pose的label映射到1080,1920上
def import_coco_tools():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        print("[ERROR] pycocotools is required for OKS mAP evaluation.")
        print("        Install with: pip install pycocotools")
        raise
    return COCO, COCOeval
    
def compute_letterbox_params(orig_w: int, orig_h: int, target_w: int, target_h: int) -> Tuple[float, float, float]:
    """
    Compute scale and pad to letterbox (keep aspect) from (orig_w, orig_h) to target canvas (target_w, target_h).
    Returns (scale, pad_w, pad_h) where pad_* are the left/top padding in pixels.
    """
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = orig_w * scale, orig_h * scale
    pad_w = (target_w - new_w) / 2.0
    pad_h = (target_h - new_h) / 2.0
    return scale, pad_w, pad_h
    
def map_bbox_xywh_to_target(bbox_xywh: np.ndarray, scale: float, pad_w: float, pad_h: float) -> np.ndarray:
    """
    bbox_xywh: [x, y, w, h] in original coordinates.
    """
    x, y, w, h = bbox_xywh.astype(np.float32)
    x = x * scale + pad_w
    y = y * scale + pad_h
    w = w * scale
    h = h * scale
    return np.array([x, y, w, h], dtype=np.float32)

def map_points_to_target(xy: np.ndarray, scale: float, pad_w: float, pad_h: float) -> np.ndarray:
    """
    xy: [..., 2] points in original image coordinates.
    Returns points mapped to target canvas using (scale, pad_w, pad_h).
    """
    out = xy.copy().astype(np.float32)
    out[..., 0] = out[..., 0] * scale + pad_w
    out[..., 1] = out[..., 1] * scale + pad_h
    return out

def build_mapped_coco_json(ann_json: str, target_w: int, target_h: int, out_json: str) -> Dict:
    """
    Load a COCO person keypoints JSON, letterbox map all keypoints + bboxes to target canvas size,
    and write a *new* JSON with updated image sizes, annotations, and areas.
    Returns the in-memory dictionary for convenience.
    """
    COCO, _ = import_coco_tools()
    coco = COCO(ann_json)

    # Build filename -> (id, orig_w, orig_h) from COCO "images"
    id_to_img = {}
    for img in coco.dataset['images']:
        id_to_img[img['id']] = img

    # Prepare new dataset dict
    new_dataset = {
        'info': coco.dataset.get('info', {}),
        'licenses': coco.dataset.get('licenses', []),
        'images': [],
        'annotations': [],
        'categories': coco.dataset.get('categories', []),
    }

    # Update images with target sizes; keep file_name and id stable
    for img in coco.dataset['images']:
        new_img = dict(img)
        # new_img['width'] = target_w
        # new_img['height'] = target_h
        new_dataset['images'].append(new_img)

    # Map annotations
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)

    for ann in tqdm(anns, desc="Mapping GT to target canvas"):
        img_info = id_to_img[ann['image_id']]
        orig_w, orig_h = img_info['width'], img_info['height']
        scale, pad_w, pad_h = compute_letterbox_params(orig_w, orig_h, target_w, target_h)

        # Keypoints: flat length 51 = 17*(x,y,v)
        kpts = np.array(ann.get('keypoints', []), dtype=np.float32).reshape(-1, 3)  # [17, 3]
        if kpts.size > 0:
            xy = kpts[:, :2]
            v = kpts[:, 2:3]  # keep original visibility (0/1/2)
            xy_m = map_points_to_target(xy, scale, pad_w, pad_h)
            kpts_m = np.concatenate([xy_m, v], axis=1).reshape(-1).tolist()
        else:
            kpts_m = ann.get('keypoints', [])

        # BBox in xywh
        bbox = np.array(ann.get('bbox', [0, 0, 0, 0]), dtype=np.float32)
        bbox_m = map_bbox_xywh_to_target(bbox, scale, pad_w, pad_h).tolist()

        # Area scales with scale^2
        area = ann.get('area', 0.0)
        area_m = float(area) * (scale ** 2)

        new_ann = dict(ann)
        new_ann['bbox'] = [float(x) for x in bbox_m]
        new_ann['area'] = float(area_m)
        new_ann['keypoints'] = [float(x) for x in kpts_m]
        new_dataset['annotations'].append(new_ann)

    # Write JSON
    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open('w', encoding='utf-8') as f:
        json.dump(new_dataset, f)

    print(f"[OK] Wrote mapped COCO JSON -> {out_json_path}")
    return new_dataset


def letterbox_mapping(size, target_w, target_h):
    """计算 letterbox 缩放 + padding 参数"""
    orig_h, orig_w = size
    r = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
    dw, dh = target_w - new_w, target_h - new_h
    dw /= 2
    dh /= 2
    return r, dw, dh

# change new ------


def setup_logger(log_name, save_dir):
    filename = '%s.log' % log_name
    save_file = os.path.join(save_dir, filename)
    # if os.path.exists(save_file):
    #     with open(save_file, "w") as log_file:
    #         log_file.truncate()
    logger.remove()
    logger.add(save_file, rotation="10 MB", format="{time} {level} {message}", level="INFO")
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.info('This is the %s log' % log_name)
    return logger


# ---------------- Postprocess: Scale back to original ----------------(box)
def scale_coords_pose(img_shape, coords, ratio, dw, dh, num_kpts=17): # 缩放
    coords[:, [0, 2]] -= dw
    coords[:, [1, 3]] -= dh
    coords[:, :4] /= ratio
    for i in range(num_kpts):
        base = 5 + i * 3
        coords[:, base] -= dw
        coords[:, base + 1] -= dh
        coords[:, base:base+2] /= ratio
        coords[:, base]     = np.clip(coords[:, base], 0, img_shape[1])
        coords[:, base + 1] = np.clip(coords[:, base + 1], 0, img_shape[0])
    return coords

def scale_coords_pose_target(target_shape, coords, ratio, dw, dh, num_kpts=17): # target_shape-->[h,w] # 扩大
    coords[:, :4] *= ratio
    coords[:, [0, 2]] += dw
    coords[:, [1, 3]] += dh
    for i in range(num_kpts):
        base = 5 + i * 3
        coords[:, base:base+2] *= ratio
        coords[:, base] += dw
        coords[:, base + 1] += dh
        coords[:, base]     = np.clip(coords[:, base], 0, target_shape[1]) # x
        coords[:, base + 1] = np.clip(coords[:, base + 1], 0, target_shape[0]) # y
    return coords


# ---------------------- Letterbox 预处理 ----------------------
def letterbox(img, new_shape=(640, 1088), color=(114, 114, 114)):
    shape = img.shape[:2]  # h,w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

# ---------------------- images load ----------------------
def load_images_files(images_path: Union[str, Path]) -> List:
    if isinstance(images_path, str):
        images_path = Path(images_path)
    assert images_path.exists()
    if images_path.is_dir():
        images = [
            i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS
        ]
    else:
        assert images_path.suffix in SUFFIXS
        images = [images_path.absolute()]
    return images

def load_image_ids(images):
    ids = []
    for image in images:
        image = os.path.basename(str(image)) # '000000320696.jpg'
        image_id = int(re.search(r'(\d+)', image).group(1)) # 320696
        ids.append(image_id)
    return ids


def process_images_fast(images: List, 
                        imgsz: Union[int, tuple]=(640, 1088),
                        batch_size: int = 64,
                        device='cuda:0'):
    if isinstance(imgsz, int):
        H, W = imgsz, imgsz
    else:
        H, W = imgsz[0], imgsz[1]

    data_list = []
    dwdh_list = []
    ratio_list = []
    draw_list = []

    # 多线程读取图片
    def read_image(path):
        img = cv2.imread(str(path))
        return img

    with ThreadPoolExecutor(max_workers=8) as executor:
        imgs_all = list(tqdm(executor.map(read_image, images), total=len(images), desc="Reading images"))

    # 分 batch 处理
    for i in tqdm(range(0, len(imgs_all), batch_size), desc="Processing batches"):
        batch_imgs = imgs_all[i:i+batch_size]
        batch_draw = [img.copy() for img in batch_imgs]

        batch_resized = []
        batch_dw_dh = []
        batch_ratio = []

        for img in batch_imgs:
            bgr, ratio, dwdh = letterbox(img, (H,W))
            batch_resized.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            batch_dw_dh.append(dwdh)
            batch_ratio.append(ratio)

        # 转为 float32 并归一化
        batch_resized = np.stack(batch_resized, axis=0).astype(np.float32) / 255.0
        batch_resized = batch_resized.transpose(0,3,1,2)  # HWC->CHW
        batch_tensor = torch.from_numpy(batch_resized)

        data_list.append(batch_tensor)
        # dwdh_list.extend([torch.tensor(dw*2, dtype=torch.float32) for dw in batch_dw_dh])
        dwdh_list.extend([torch.tensor(dw*2, dtype=torch.float32) for dw in batch_dw_dh])
        ratio_list.extend(batch_ratio)
        draw_list.extend(batch_draw)

    data_tensor = torch.cat(data_list, dim=0)  # [N,3,H,W]
    return data_tensor, dwdh_list, ratio_list, draw_list



def batch_generator_data(data: torch.tensor, batch: int):
    """
    生成器函数，每次生成一个批量数据。
    
    :param data: 数据列表
    :param batch_size: 每个批量的大小
    """
    for i in range(0, len(data), batch):
        yield data[i:i + batch]

# batch NMS


def batch_pose_postprocess(
    data: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65
):
    """
    批量后处理 (Batch B, 56, N) -> 每张图各自的 (boxes, scores, kpts)
    """
    B, _, N = data.shape
    data = data.permute(0, 2, 1).contiguous()  # (B, N, 56)
    results = []
    for b in range(B):
        p = data[b]  # (N, 56)
        bboxes, scores, kpts = p[:, :4], p[:, 4], p[:, 5:].reshape(N, 17, 3)
        # 过滤置信度
        mask = scores > conf_thres
        if mask.sum() == 0:
            results.append((torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0, 17, 3))))
            continue
        boxes, sc, kp = bboxes[mask], scores[mask], kpts[mask]
        # xywh -> xyxy
        xy, wh = boxes[:, :2], boxes[:, 2:]
        xyxy = torch.cat([xy - 0.5*wh, xy + 0.5*wh], dim=1)
        # NMS
        keep = nms(xyxy, sc, iou_thres)
        # print(xyxy[keep].shape)
        results.append((xyxy[keep], sc[keep], kp[keep]))
    return results

@torch.inference_mode() # 速度比上一个慢，上一个是2.4ms,这一个是4.5ms
def batch_pose_postprocess_fast(
    data: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65
):
    """
    超快批量后处理 (GPU 向量化 + batched_nms + 无循环)
    输入: data [B, 56, N]
    输出: List[ (boxes, scores, kpts) * B ]
    """
    device = data.device
    B, _, N = data.shape
    data = data.permute(0, 2, 1).contiguous()  # [B, N, 56]
    
    # --------------------------
    # 1. 批量提取boxes / scores / kpts
    # --------------------------
    bboxes = data[..., :4]          # [B, N, 4] xywh
    scores = data[..., 4]           # [B, N]
    kpts = data[..., 5:].reshape(B, N, 17, 3)  # [B, N, 17, 3]
    
    # --------------------------
    # 2. 批量 mask 筛选 (GPU)
    # --------------------------
    mask = scores > conf_thres      # [B, N]
    valid_idx = mask.nonzero(as_tuple=False)  # [M, 2], 每行 [batch_idx, point_idx]
    
    if valid_idx.numel() == 0:
        return [(torch.zeros((0, 4), device=device),
                 torch.zeros((0,), device=device),
                 torch.zeros((0, 17, 3), device=device)) for _ in range(B)]
    
    # --------------------------
    # 3. 批量展开有效数据
    # --------------------------
    batch_ids = valid_idx[:, 0]     # [M]
    point_ids = valid_idx[:, 1]     # [M]
    
    valid_boxes = bboxes[batch_ids, point_ids]  # [M, 4]
    valid_scores = scores[batch_ids, point_ids] # [M]
    valid_kpts = kpts[batch_ids, point_ids]    # [M, 17, 3]
    
    # --------------------------
    # 4. xywh -> xyxy (GPU)
    # --------------------------
    xy = valid_boxes[:, :2]
    wh = valid_boxes[:, 2:]
    xyxy = torch.cat([xy - 0.5 * wh, xy + 0.5 * wh], dim=1)  # [M, 4]
    
    # --------------------------
    # 5. 批量NMS (torchvision batched_nms)
    # --------------------------
    keep = batched_nms(xyxy, valid_scores, batch_ids, iou_thres)  # [K]
    xyxy, valid_scores, valid_kpts, batch_ids = xyxy[keep], valid_scores[keep], valid_kpts[keep], batch_ids[keep]
    
    # --------------------------
    # 6. 拆分回每张图片
    # --------------------------
    results = []
    for b in range(B):
        idx = (batch_ids == b)
        results.append((xyxy[idx], valid_scores[idx], valid_kpts[idx]))
    
    return results


# evaluate for single img infer and post process 
def evaluate_trt(engine_path, coco_img_dir, coco_anno_path, ):
    trt_model = TRTModule(engine_path, device="cuda:0")
    H, W = trt_model.H, trt_model.W
    coco_gt = COCO(coco_anno_path)
    results = []

    img_ids = coco_gt.getImgIds()
    for img_id in tqdm(img_ids, desc="Evaluating"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(coco_img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path) # bgr

        # preprocess
        img_input, ratio, dw, dh = letterbox(img, (H, W))
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB) # RGB格式(H, W, 3)
        img_input = img_input[:, :, ::-1].transpose(2, 0, 1)
        img_input = np.ascontiguousarray(img_input, dtype=np.float16) / 255.0 # [3, H, W]

        # inference
        img_input = torch.tensor(img_input).unsqueeze(0) # [1, 3, H, W]
        pred, _ = trt_model(img_input) # [1, 56, num_boxes] # numpy
        if pred.shape[-1] == 0:
            continue
        
        # NMS
        boxes, scores, kpts = batch_pose_postprocess(torch.tensor(pred))[0] # [](xyxy, sc, kp) #([N,4], [N], [N,51])
        
        # boxes, scores, kpts = result
        scores = scores.unsqueeze(1) # [N,1]
        kpts = kpts.reshape(-1, 17*3)
        # print(result[0].shape)
        # print(result[1].shape)
        # print(result[2].shape)
        result = torch.cat([boxes, scores, kpts], dim=1).numpy()# [N,56]

        # scale back to original image
        pred = scale_coords_pose(img.shape, result, ratio, dw, dh, num_kpts=17)

        # save results in COCO format
        for p in pred:
            x1, y1, x2, y2 = p[:4]
            score = p[4]
            kpts = p[5:].reshape(17, 3)
            # results.append({
            #     "image_id": img_id,
            #     "category_id": 1,  # human pose
            #     "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            #     "score": float(score),
            #     "keypoints": keypoints.flatten().tolist()
            # })
           # 关键点转换为 COCO 格式
            coco_kpts = []
            for (x, y, conf) in kpts:
                v = 2 if conf > 0.25 else 1  # >0.5视为可见，否则不可见
                coco_kpts.extend([float(x), float(y), int(v)])
        
            results.append({
                "image_id": int(img_id),
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
                "keypoints": coco_kpts
            })

    # save predictions
    import json
    res_file = "trt_predictions.json"
    with open(res_file, "w") as f:
        json.dump(results, f)
    print(f"trt prediction results saved to->{res_file}")

    # COCO mAP evaluation
    coco_dt = coco_gt.loadRes(res_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    
    
# evaluate for batch img infer and post process
def batch_evaluate_trt(engine_path, coco_img_dir, coco_anno_path, batch=10, log_name='trt_evaluate',
                       target_size=(1080,1920)):
    # logger
    # 获取文件名（带后缀）
    filename_with_ext = os.path.basename(engine_path)  # 'yolov8s-pose-prune-sp0.5-op13-h640w1088-dynamic.engine'

    # 去掉后缀
    filename = os.path.splitext(filename_with_ext)[0]  # 'yolov8s-pose-prune-sp0.5-op13-h640w1088-dynamic'
    save_dir = os.path.join("output", filename)
    
    logger = setup_logger(log_name, save_dir)
    
    
    # engine load
    trt_model = TRTModule(engine_path, device="cuda:0")
    H, W = trt_model.H, trt_model.W
    # print(H,W)
    # H, W = 640, 640
    # coco gt
    coco_gt = COCO(coco_anno_path)
    
    image_files = load_images_files(coco_img_dir)
    image_ids = load_image_ids(image_files)
    N = len(image_files)
    # batch inputs-->preprocess
    data_tensor, dwdh_list, ratio_list, draw_list = process_images_fast(image_files, imgsz=(H,W))
    # 要计算新

    # batch iters
    data_tensor_iters = batch_generator_data(data_tensor, batch)
    image_ids_iters = batch_generator_data(image_ids, batch)
    dwdh_iters = batch_generator_data(dwdh_list, batch)
    ratio_iters = batch_generator_data(ratio_list, batch)
    draw_iters = batch_generator_data(draw_list, batch)
    
    results = []
    # batch trt-->infer
    infer_time = []
    for batch_data in tqdm(data_tensor_iters): # [batch, 3, H, W]
        # batch infer 
        pred, trt_time = trt_model(batch_data)  # [batch, 56, N]
        results.append(pred)
        if isinstance(trt_time, List): # Test
            trt_time = sum(trt_time) / len(trt_time)
        infer_time.append(trt_time)
    logger.info(f"TRT INFER {N} IMAGES->Total: {sum(infer_time):.3f}ms")
    logger.info(f"TRT INFER {N} IMAGES->AVERAGE: {sum(infer_time)/N:.3f}ms/img")
    
    # ALL Results-->post process
    
    post_time = []
    post_results = []
    for batch_idx, result in tqdm(enumerate(results), desc="Post Processing TRT Results"):
        # batch_image_ids = image_ids_iters[batch_idx]  # [batch,]
        
        # batch-->post process
        # NMS
        # boxes, scores, kpts = batch_pose_postprocess(torch.tensor(result))[0] # [](xyxy, sc, kp) #([N,4], [N], [N,51],...)
        
        start = time.time()
        # result = batch_pose_postprocess_fast(torch.tensor(result)) # [(boxes, scores, kpts)*batch]-->
        # reulst= batch_pose_postprocess_ultra(torch.tensor(result))
        pose_reulst = batch_pose_postprocess(torch.tensor(result))
        post_results.append(pose_reulst)
        batch_post_time = (time.time()-start) * 1000
        post_time.append(batch_post_time)
        
    logger.info(f"TRT POST {N} IMAGES->Total: {sum(post_time):.3f}ms")
    logger.info(f"TRT POST {N} IMAGES->AVERAGE: {sum(post_time)/N:.3f}ms/img")
    
    # Result Evaluate
    all_results = []
    for batch_idx, preds in tqdm(enumerate(post_results)): # [(boxes, scores, kpts)*batch]
        
        image_ids = next(iter(image_ids_iters)) # [batch]
        batch_dwdh = next(iter(dwdh_iters))
        batch_ratio = next(iter(ratio_iters))
        for idx, (boxes, scores, kpts) in enumerate(preds):
            img_id = image_ids[idx]
            dw, dh = batch_dwdh[idx][0].numpy(), batch_dwdh[idx][1].numpy()
            ratio = batch_ratio[idx]
            img_info = coco_gt.loadImgs(img_id)[0]
            h = img_info['height'] # 1080, assert this is org img h
            w = img_info['width']  # 1920, assert this is org img h
            if target_size is not None:
                # 更新映射dw, dh, ratio
                # ratio, dw, dh = compute_letterbox_params(W, H, target_size[1], target_size[0])
                # scale, pad_w, pad_h = compute_letterbox_params(W, H, target_size[1], target_size[0])
                # h, w = target_size
                # ratio, dw, dh = compute_letterbox_params(target_size[1], target_size[0],W, H)
                scale, pad_w, pad_h = compute_letterbox_params(w, h, target_size[1], target_size[0])
            # scale back to original image
            scores = scores.unsqueeze(1) # [M,1]
            kpts = kpts.reshape(-1, 17*3)
            result = torch.cat([boxes, scores, kpts], dim=1).numpy()# [M,56]
            # # scale back to original image
            pred = scale_coords_pose((h,w), result, ratio, dw, dh, num_kpts=17) # 映射到原始图片
            # if target_size is not None:
            #     h, w = target_size
            #     pred = scale_coords_pose_target((h,w), result, scale, pad_w, pad_h, num_kpts=17) # 映射到目标size
            # else:
            #      pred = scale_coords_pose((h,w), result, ratio, dw, dh, num_kpts=17) # 映射到原始图片
            # save results in COCO format
            for p in pred:
                x1, y1, x2, y2 = p[:4]
                if target_size is not None:
                    x1, x2 = x1*scale + pad_w, x1*scale + pad_w
                    y1, y2 = y1*scale + pad_h, y2*scale + pad_h
                
                score = p[4]
                kpts = p[5:].reshape(17, 3)
            # 关键点转换为 COCO 格式
                coco_kpts = []
                for (x, y, conf) in kpts:
                    v = 2 if conf > 0.25 else 1  # >0.25视为可见，否则不可见
                    if target_size is not None:
                        x, y = x*scale + pad_w, y*scale + pad_h
                    coco_kpts.extend([float(x), float(y), int(v)])
                    
                all_results.append({
                    "image_id": int(img_id),
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], # xywh
                    "score": float(score),
                    "keypoints": coco_kpts
                })
                
    # save predictions
    import json
    res_file = os.path.join(save_dir, "trt_predictions.json")
    with open(res_file, "w") as f:
        json.dump(all_results, f)
    logger.info(f"trt prediction results saved to->{res_file}")

    # COCO mAP evaluation
    coco_dt = coco_gt.loadRes(res_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # 3. 获取 mAP50 (keypoints)
    map50_pose = coco_eval.stats[1]  # stats[1] 是 keypoints mAP@0.5
    logger.info(f"Keypoints mAP@0.5:{map50_pose}")
    
def trt_predict(engine_path, image_path, coco_anno_path, batch=10, log_name='trt_predict'):
    save_dir = os.path.join("output", os.path.basename(engine_path).split('.')[0])
    logger = setup_logger(log_name, save_dir)
    # engine load
    trt_model = TRTModule(engine_path, device="cuda:0")
    H, W = trt_model.H, trt_model.W
    
    # coco gt
    coco_gt = COCO(coco_anno_path)
    
    image_files = load_images_files(image_path) # List
    image_ids = load_image_ids(image_files)
    N = len(image_files)
    
    data_tensor, dwdh_list, ratio_list, draw_list = process_images_fast(image_files, imgsz=(H,W))
    # batch iters
    data_tensor_iters = batch_generator_data(data_tensor, batch)
    image_ids_iters = batch_generator_data(image_ids, batch)
    dwdh_iters = batch_generator_data(dwdh_list, batch)
    ratio_iters = batch_generator_data(ratio_list, batch)
    draw_iters = batch_generator_data(draw_list, batch)
    
    results = []
    # batch trt-->infer
    infer_time = []
    for batch_data in tqdm(data_tensor_iters): # [batch, 3, H, W]
        # batch infer 
        pred, trt_time = trt_model(batch_data)  # [batch, 56, N]
        results.append(pred)
        if isinstance(trt_time, List): # Test
            trt_time = sum(trt_time) / len(trt_time)
        infer_time.append(trt_time)
    logger.info(f"TRT INFER {N} IMAGES->Total: {sum(infer_time):.3f}ms")
    logger.info(f"TRT INFER {N} IMAGES->AVERAGE: {sum(infer_time)/N:.3f}ms/img")
    
    # ALL Results-->post process
    
    post_time = []
    post_results = []
    for batch_idx, result in tqdm(enumerate(results), desc="Post Processing TRT Results"):
        # batch_image_ids = image_ids_iters[batch_idx]  # [batch,]
        
        # batch-->post process
        # NMS
        # boxes, scores, kpts = batch_pose_postprocess(torch.tensor(result))[0] # [](xyxy, sc, kp) #([N,4], [N], [N,51],...)
        start = time.time()
        # result = batch_pose_postprocess_fast(torch.tensor(result)) # [(boxes, scores, kpts)*batch]-->
        # reulst= batch_pose_postprocess_ultra(torch.tensor(result))
        pose_reulst = batch_pose_postprocess(torch.tensor(result))
        post_results.append(pose_reulst)
        batch_post_time = (time.time()-start) * 1000
        post_time.append(batch_post_time)
        
    logger.info(f"TRT POST {N} IMAGES->Total: {sum(post_time):.3f}ms")
    logger.info(f"TRT POST {N} IMAGES->AVERAGE: {sum(post_time)/N:.3f}ms/img")
    
    # Result Evaluate
    all_results = []
    for batch_idx, preds in tqdm(enumerate(post_results)): # [(boxes, scores, kpts)*batch]
        
        image_ids = next(iter(image_ids_iters)) # [batch]
        batch_draw = next(iter(draw_iters))
        batch_dwdh = next(iter(dwdh_iters))
        batch_ratio = next(iter(ratio_iters))
        for idx, (boxes, scores, kpts) in enumerate(preds):
            img_id = image_ids[idx]
            draw = batch_draw[idx]
            dw, dh = batch_dwdh[idx][0].numpy(), batch_dwdh[idx][1].numpy()
            ratio = batch_ratio[idx]
            
            img_info = coco_gt.loadImgs(img_id)[0]
            h = img_info['height']
            w = img_info['width']
            
            file_name = img_info['file_name']
            save_img_name = file_name.split('.')[0]+f"_trt_prediction.jpg"
            save_img = os.path.join(save_dir, save_img_name)
            
            scores = scores.unsqueeze(1) # [M,1]
            kpts = kpts.reshape(-1, 17*3)
            result = torch.cat([boxes, scores, kpts], dim=1).numpy()# [M,56]
            # scale back to original image
            pred = scale_coords_pose((h,w), result, ratio, dw, dh, num_kpts=17)
            for p in pred:
                bbox = p[:4].round().tolist()
                bbox = [int(x) for x in bbox]
                score = p[4]
                kpt = p[5:].reshape(17,3)
                color = COLORS['person']
                cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
                cv2.putText(draw,
                            f'person:{score:.3f}', (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, [225, 255, 255],
                            thickness=2)
                for i in range(19):
                    if i < 17:
                        px, py, ps = kpt[i]
                        
                        if ps > 0.5:
                            kcolor = KPS_COLORS[i]
                            px = round(float(px))
                            py = round(float(py))
                            cv2.circle(draw, (px, py), 5, kcolor, -1)
                    xi, yi = SKELETON[i]
                    pos1_s = kpt[xi - 1][2]
                    pos2_s = kpt[yi - 1][2]
                    if pos1_s > 0.5 and pos2_s > 0.5:
                        limb_color = LIMB_COLORS[i]
                        # pos1_x = round(float(kpt[xi - 1][0] - dw) / ratio)
                        # pos1_y = round(float(kpt[xi - 1][1] - dh) / ratio)

                        # pos2_x = round(float(kpt[yi - 1][0] - dw) / ratio)
                        # pos2_y = round(float(kpt[yi - 1][1] - dh) / ratio)
                        pos1_x = round(float(kpt[xi - 1][0]))
                        pos1_y = round(float(kpt[xi - 1][1]))

                        pos2_x = round(float(kpt[yi - 1][0]))
                        pos2_y = round(float(kpt[yi - 1][1]))

                        cv2.line(draw, (pos1_x, pos1_y), (pos2_x, pos2_y),
                                limb_color, 2)
                        
            cv2.imwrite(str(save_img), draw)
            # cv2.imshow('result', draw)
            # cv2.waitKey(0)
            # cv2.imwrite(str(save_img), draw)
                
                

    
if __name__ == "__main__":
    os.chdir(BASE_DIR)
    # 映射json
    # ann_json = "datasets/coco-pose/annotations/person_keypoints_val2017.json"
    # target_w = 1920
    # target_h = 1088
    # save_json = "datasets/coco-pose/annotations/person_keypoints_val2017_h1080w1920_new.json"
    # build_mapped_coco_json(ann_json, target_w, target_h, save_json)
    
    # engine_path = 'weights/yolov8s-pose-prune-sp0.5-distill-dynamic-shape.engine'
    # engine_path = 'weights/yolov8s-pose-op13-h640w1088-dynamic.engine' # base
    # engine_path = 'weights/yolov8s-pose-op13-h640w640-dynamic-fp16.engine'
    # engine_path = "weights/yolov8s-pose-qat-h640w1088.engine"
    # engine_path = "weights/yolov8s-pose-qat.engine" # hw640
    # engine_path = "weights/yolov8s-pose-prune-sp0.3-h640w1088-fp16.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.3-h640w640-fp16.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.4-op13-h640w1088-dynamic-fp16.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.7-h640w640-fp16.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.5-op13-h640w1088-dynamic.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.5-h640w640-fp16.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.5-distill-dynamic-shape.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.5-distill-train640-h640w1088.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.5-distill-train640-h640w640.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.6-distill-train640_1088-h640w1088.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.7-distill-train640_1088-h640w1088.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.5-distill-train640_1088-h640w640.engine"
    engine_path = "weights/yolov8s-pose-qat.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.6-op13-h640w1088-dynamic-fp16.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.6-h640w640-fp16.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.7-op13-h640w1088-dynamic-fp16.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.7-h640w640-fp16.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.5-distill-h640w1088-fp16.engine"
    # engine_path = "weights/yolov8s-pose-prune-sp0.6-distill-h640w640-fp16.engine"
    coco_img_dir = "datasets/coco-pose/images/val2017"
    
    # coco_anno_path = "datasets/coco-pose/annotations/person_keypoints_val2017.json"
    # coco_anno_path = "datasets/coco-pose/annotations/person_keypoints_val2017_h1080w1920.json" # hw-->1080,1920
    coco_anno_path = "datasets/coco-pose/annotations/person_keypoints_val2017_h1080w1920_new.json" # hw-->org
    # coco_anno_path = "./annotations/person_keypoints_mapped_1080x1920.json"
    
    batch= 10
    # # trt_predict(engine_path, image_path="./images", coco_anno_path=coco_anno_path, batch=10, log_name='trt_predict')
    # batch_evaluate_trt(engine_path, coco_img_dir, coco_anno_path, batch, log_name='trt_evaluate', 
    #                    target_size=None)
    batch_evaluate_trt(engine_path, coco_img_dir, coco_anno_path, batch, log_name='trt_evaluate_base', 
                       target_size=(1080,1920))
    
    
