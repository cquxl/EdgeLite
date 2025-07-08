
from ultralytics import YOLO, __version__
from ultralytics.utils import LOGGER, RANK
from ultralytics.cfg import check_cfg
import yaml  # 替代被移除的yaml_load
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from matplotlib import pyplot as plt
from copy import deepcopy
from pathlib import Path
from typing import List, Union

from ultralytics.nn.modules import C2f, Conv, Detect, Pose, Bottleneck
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import initialize_weights, de_parallel
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import attempt_load_one_weight

# 如果没有官方暴露的 TASK_MAP，就从任务模块中手动构建
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.models.yolo.classify import ClassificationTrainer

import os
import d2l.torch as d2l
d2l.use_svg_display()

TASK_MAP = {
    "detect": (YOLO, DetectionTrainer),
    "segment": (YOLO, SegmentationTrainer),
    "classify": (YOLO, ClassificationTrainer),
    "pose": (YOLO, PoseTrainer),
}


def save_model_v2(self: BaseTrainer):
    """
    Disabled half precision saving. originated from ultralytics/yolo/engine/trainer.py
    """
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(de_parallel(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt


def final_eval_v2(self: BaseTrainer):
    """
    模型最终验证（适配Ultralytics 8.3.149）
    """
    for f in [self.last, self.best]:
        if f.exists():
            strip_optimizer_v2(f)  # 剥离优化器
            if f == self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.metrics = self.validator(model=str(f))
                self.metrics.pop('fitness', None)
                if hasattr(self, 'run_callbacks'):
                    self.run_callbacks('on_fit_epoch_end')

def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    Disabled half precision saving. originated from ultralytics/yolo/utils/torch_utils.py
    """
    x = torch.load(f, map_location=torch.device('cpu'))
    args = {**DEFAULT_CFG_DICT, **x['train_args']}  # combine model args with default args, preferring model args
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'ema', 'updates':  # keys
        x[k] = None
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


# def train_v2(self, pruning=False, **kwargs):
#     """
#     剪枝训练模式适配（解决TASK_MAP和yaml_load问题）
#     """
#     # 检查PyTorch模型
#     self._check_is_pytorch_model()
#
#     # 处理HUB会话参数
#     if self.session and kwargs:
#         LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local arguments.')
#         kwargs = self.session.train_args
#
#     # 合并配置覆盖项
#     overrides = self.overrides.copy()
#     overrides.update(kwargs)
#
#     # 加载外部配置文件
#     if kwargs.get('cfg'):
#         cfg_path = check_yaml(kwargs['cfg'])  # 路径校验
#         with open(cfg_path, errors='ignore') as f:
#             overrides = yaml.safe_load(f)  # 使用yaml.safe_load替代yaml_load[1](@ref)
#         LOGGER.info(f"cfg file passed. Overriding default params with {cfg_path}")
#
#     # 设置训练模式和必要参数
#     overrides['mode'] = 'train'
#     if not overrides.get('data'):
#         raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
#     if overrides.get('resume'):
#         overrides['resume'] = self.ckpt_path
#
#     # 动态加载训练器（替代TASK_MAP）[2](@ref)
#     self.task = overrides.get('task') or self.task
#     TrainerClass = TASK_MAP[self.task][1]
#     self.trainer = TrainerClass(overrides=overrides, _callbacks=self.callbacks)
#
#     # ===== 剪枝模式特殊处理 =====
#     if pruning:
#         LOGGER.info("🔧 Pruning mode activated — skipping get_model.")
#         self.trainer.pruning = True
#         self.trainer.model = self.model
#         # 替换方法以禁用半精度
#         self.trainer.save_model = save_model_v2.__get__(self.trainer)
#         self.trainer.final_eval = final_eval_v2.__get__(self.trainer)
#     else:
#         if not overrides.get('resume'):
#             # 非剪枝模式加载初始权重
#             self.trainer.model = self.trainer.get_model(
#                 weights=self.model if self.ckpt else None,
#                 cfg=self.model.yaml
#             )
#             self.model = self.trainer.model
#
#     # 关联HUB会话并开始训练
#     self.trainer.hub_session = self.session
#     self.trainer.train()
#
#     # 训练后更新模型
#     if RANK in (-1, 0):
#         self.model, _ = attempt_load_one_weight(str(self.trainer.best))
#         self.overrides = self.model.args
#         self.metrics = getattr(self.trainer.validator, 'metrics', None)


def train_v2(self: YOLO, pruning=True, **kwargs):
    """
    Disabled loading new model when pruning flag is set. originated from ultralytics/yolo/engine/model.py
    """

    self._check_is_pytorch_model()
    if self.session:  # Ultralytics HUB session
        if any(kwargs):
            LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args
    overrides = self.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get('cfg'):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = yaml_load(check_yaml(kwargs['cfg']))
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)

    if not pruning:
        if not overrides.get('resume'):  # manually set model only if not resuming
            LOGGER.info("🔧 downloading model.")
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

    else:
        # pruning mode
        LOGGER.info("🔧 Pruning mode activated — skipping get_model.")
        self.trainer.pruning = True
        self.trainer.model = self.model
        device = self.device
        print(f'self device: {self.device}')
        if self.device != kwargs.get('device'):
            device = kwargs.get('device')
            print(f'device: {device }')
        self.trainer.model = self.trainer.model.to(device)
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train()
    # Update model and cfg after training
    if RANK in (-1, 0):
        LOGGER.info("🔧 Rank downloading model.")
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.model.to(self.device)
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)

def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)

class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add


def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)

def save_pruning_performance_graph(x, y1, y2, y3):
    """
    Draw performance change graph
    Parameters
    ----------
    x : List
        Parameter numbers of all pruning steps
    y1 : List
        mAPs after fine-tuning of all pruning steps
    y2 : List
        MACs of all pruning steps
    y3 : List
        mAPs after pruning (not fine-tuned) of all pruning steps

    Returns
    -------

    """
    try:
        plt.style.use("ggplot")
    except:
        pass

    x, y1, y2, y3 = np.array(x), np.array(y1), np.array(y2), np.array(y3)
    y2_ratio = y2 / y2[0]

    # create the figure and the axis object
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the pruned mAP and recovered mAP
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('mAP')
    ax.plot(x, y1, label='recovered mAP')
    ax.scatter(x, y1)
    ax.plot(x, y3, color='tab:gray', label='pruned mAP')
    ax.scatter(x, y3, color='tab:gray')

    # create a second axis that shares the same x-axis
    ax2 = ax.twinx()

    # plot the second set of data
    ax2.set_ylabel('MACs')
    ax2.plot(x, y2_ratio, color='tab:orange', label='MACs')
    ax2.scatter(x, y2_ratio, color='tab:orange')

    # add a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    ax.set_xlim(105, -5)
    ax.set_ylim(0, max(y1) + 0.05)
    ax2.set_ylim(0.05, 1.05)

    # calculate the highest and lowest points for each set of data
    max_y1_idx = np.argmax(y1)
    min_y1_idx = np.argmin(y1)
    max_y2_idx = np.argmax(y2)
    min_y2_idx = np.argmin(y2)
    max_y1 = y1[max_y1_idx]
    min_y1 = y1[min_y1_idx]
    max_y2 = y2_ratio[max_y2_idx]
    min_y2 = y2_ratio[min_y2_idx]

    # add text for the highest and lowest values near the points
    ax.text(x[max_y1_idx], max_y1 - 0.05, f'max mAP = {max_y1:.2f}', fontsize=10)
    ax.text(x[min_y1_idx], min_y1 + 0.02, f'min mAP = {min_y1:.2f}', fontsize=10)
    ax2.text(x[max_y2_idx], max_y2 - 0.05, f'max MACs = {max_y2 * y2[0] / 1e9:.2f}G', fontsize=10)
    ax2.text(x[min_y2_idx], min_y2 + 0.02, f'min MACs = {min_y2 * y2[0] / 1e9:.2f}G', fontsize=10)

    plt.title('Comparison of mAP and MACs with Pruning Ratio')
    plt.savefig('pruning_perf_change.png')


def plot_pruning_performance(sparsity_list, macs_list, nparams_list,
                             pruned_map_list, map_list, speedup_list,
                             title="Model Pruning Performance Analysis",
                             figsize=(10, 4),
                             save_dir=None):
    """
    绘制剪枝模型的性能对比图（双Y轴布局）

    参数:
        sparsity_list   : 稀疏度列表 (x轴)
        macs_list       : MACs变化列表 (百分比)
        nparams_list    : 参数量变化列表 (百分比)
        pruned_map_list : 剪枝后mAP50列表
        map_list        : 微调后mAP50列表
        speedup_list    : 加速比列表
        title           : 图表标题 (默认"Model Pruning Performance Analysis")
        figsize         : 图表尺寸 (默认(16,6))
    """
    # 创建画布与子图布局
    sparsity_list, macs_list, nparams_list = np.array(sparsity_list), np.array(macs_list), np.array(nparams_list)
    pruned_map_list, map_list, speedup_list = np.array(pruned_map_list), np.array(map_list), np.array(speedup_list)
    macs_list = macs_list / 1e9 # G
    nparams_list = nparams_list / 1e6 # M
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # --- 子图1：计算量与参数量关系 ---
    ax1.set_xlabel('Sparsity (Pruning Ratio)')
    ax1.set_ylabel('MACs (G)', color='tab:blue')
    ax1.plot(sparsity_list, macs_list, 'o-', color='tab:blue', label='MACs')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 参数量Y轴
    ax1_r = ax1.twinx()
    ax1_r.set_ylabel('Params (M)', color='tab:red')
    ax1_r.plot(sparsity_list, nparams_list, 's--', color='tab:red', label='Params')
    ax1_r.tick_params(axis='y', labelcolor='tab:red')
    ax1_r.set_ylim(0, max(nparams_list) * 1.1)  # 自动调整坐标范围

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # --- 子图2：精度与加速比关系 ---
    ax2.set_xlabel('Sparsity (Pruning Ratio)')
    ax2.set_ylabel('mAP50', color='tab:green')
    ax2.plot(sparsity_list, pruned_map_list, 'D-', color='tab:purple', label='Pruned mAP50')
    ax2.plot(sparsity_list, map_list, '^-', color='tab:green', label='Fine-tuned mAP50')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 加速比Y轴
    ax2_r = ax2.twinx()
    ax2_r.set_ylabel('Speedup', color='tab:orange')
    ax2_r.plot(sparsity_list, speedup_list, '*-', color='tab:orange', label='Speedup')
    ax2_r.tick_params(axis='y', labelcolor='tab:orange')
    ax2_r.set_ylim(min(speedup_list) * 0.9, max(speedup_list) * 1.1)  # 自动调整范围

    # 合并图例
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='best')

    # --- 全局调整与显示 ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留空间
    if save_dir:
        plt.savefig(os.path.join(save_dir,'pruning_performance_change.png'), dpi=300, bbox_inches='tight')





