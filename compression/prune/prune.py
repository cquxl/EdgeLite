import torch
import math
import os
from copy import deepcopy
from ultralytics.nn.tasks import PoseModel
import torch_pruning as tp

from .utils import save_model_v2, final_eval_v2, strip_optimizer_v2, train_v2, replace_c2f_with_c2f_v2, save_pruning_performance_graph, plot_pruning_performance

from ultralytics import YOLO
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics.nn.modules import Detect, Pose


class YOLOv8PosePrune:
    def __init__(self, args, yolo_cfg):
        self.args = args
        self.cfg = yolo_cfg

        self.weight = args.weight
        self.device = args.device

        self.logger = args.logger
        self.batch_size = args.batch_size

        self.iterative_steps = args.iterative_steps
        self.target_prune_rate = args.target_prune_rate
        self.p = self.args.p

        self.example_inputs = torch.randn(1, 3, self.cfg.imgsz, self.cfg.imgsz).to(self.device)

        self.model = self.load_model()
        self.init_metrics()
        self.validation_org_model()



    def load_model(self):
        self.logger.info(f'Load model from {self.weight}')
        model = YOLO(self.weight)
        # ✅ 添加防止 fallback 的关键字段
        model.pt_path = self.weight
        if not hasattr(model, 'args'):
            model.args = {"model": self.weight}  # 防止 AMP fallback
        model.__setattr__("train_v2", train_v2.__get__(model))
        model.model.train()
        replace_c2f_with_c2f_v2(model.model)
        initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace
        for name, param in model.model.named_parameters():
            param.requires_grad = True
        model.to(self.device)
        self.logger.info(f'self model device:{model.device}')
        return model

    def init_metrics(self):
        """
        x: sparsity_list-->each iter has its own actual sparsity = 1 - nparams_list / base_nparams
        y1: MACs-->macs_list
        y2: nparams (model parameters)-->nparams_list

        y3: pruned_mAP-->pruned_map_list
        y4: recovered (fine tuned)_mAP-->map_list
        y5: speedup-->speed_list

        :return:
        """
        self.macs_list, self.nparams_list, self.map_list, self.pruned_map_list = [], [], [], []
        self.sparsity_list = [0.0]  # x, 1- nparams_list / base_nparams
        self.speedup_list = [1.0]     # base_speed / pruned_speed

    def validation_org_model(self):
        self.base_macs, self.base_nparams = tp.utils.count_ops_and_params(self.model.model, self.example_inputs)
        # do validation before pruning model
        self.cfg.name= f"baseline_val"
        self.cfg.batch = 1
        self.validation_model = deepcopy(self.model)
        metrics = self.validation_model.val(**vars(self.cfg))   # dict input
        self.base_speed = metrics.speed['inference']
        self.init_map = metrics.pose.map50

        self.macs_list.append(self.base_macs)
        # self.nparams_list.append(100)
        self.nparams_list.append(self.base_nparams)
        self.map_list.append(self.init_map)
        self.pruned_map_list.append(self.init_map)

        self.logger.info(f"box map50-95:{metrics.box.map}, box map50:{metrics.box.map50}"
                         f"pose map50-95:{metrics.pose.map}, pose map50:{metrics.pose.map50}")
        self.logger.info(f"yolov8s-pose dense model->speed:{metrics.speed}")
        self.logger.info(f"yolov8s-pose dense model: MACs={self.base_macs / 1e9: .5f} G, #Params={self.base_nparams / 1e6: .5f} M, mAP={self.init_map: .5f}")

    def prune(self):
        pruning_ratio = 1 - math.pow((1 - self.target_prune_rate), 1 / self.iterative_steps) # 0.03
        # pruning_ratio = self.target_prune_rate
        for i in range(self.iterative_steps):
            self.model.to(self.device)  # make sure second pruning model from trainer model device -->cuda
            self.logger.info(f"Pruning steps: {i+1}/{self.iterative_steps}")
            self.model.model.train()
            for name, param in self.model.model.named_parameters():
                param.requires_grad = True
            ignored_layers = []
            unwrapped_parameters = []
            for m in self.model.model.modules():
                if isinstance(m, (Detect,Pose)):
                    ignored_layers.append(m)

            self.pruner = tp.pruner.GroupNormPruner(
                self.model.model,
                self.example_inputs,
                importance=tp.importance.GroupMagnitudeImportance(p=self.p),  # L2 norm pruning,
                iterative_steps=1,
                pruning_ratio=pruning_ratio,
                ignored_layers=ignored_layers,
                unwrapped_parameters=unwrapped_parameters
            )
            self.pruner.step()
            # pre fine-tuning validation
            self.cfg.name = f"pruning_step_{i}_pre_val"
            self.cfg.batch = 1
            self.validation_model.model = deepcopy(self.model.model)

            metrics = self.validation_model.val(**vars(self.cfg))
            pruned_map = metrics.pose.map50
            pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(self.pruner.model, self.example_inputs)

            # current_speed_up = float(self.macs_list[0]) / pruned_macs
            # pruned_speed = metrics.speed['inference']
            # current_speed_up = self.base_speed / pruned_speed
            # self.speedup_list.append(current_speed_up )

            self.macs_list.append(pruned_macs)
            # self.nparams_list.append(pruned_nparams / self.base_nparams * 100)
            self.nparams_list.append(pruned_nparams)
            self.pruned_map_list.append(pruned_map)

            pruned_sparsity = 1 - pruned_nparams / self.base_nparams
            self.sparsity_list.append(pruned_sparsity)

            self.logger.info(f'target sparsity{self.target_prune_rate}->step {i} pruning ratio {pruning_ratio}, '
                        f'current actual sparsity:{pruned_sparsity} ')
            self.logger.info(f"step {i} yolov8s-pose pruned model->speed:{metrics.speed}")

            self.logger.info(f"step {i} pruned model->box map50-95:{metrics.box.map}, box map50:{metrics.box.map50}"
                             f"pose map50-95:{metrics.pose.map}, pose map50:{metrics.pose.map50}")

            self.logger.info(f"After pruning iter {i}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
                  f"mAP={pruned_map}")

            # fine-tune your model here
            if self.args.fine_tune:
                self.logger.info(f"Start fine-tuning iter {i}")
                self.fine_tune(i)
                if self.init_map - self.current_map > self.args.max_map_drop:
                    self.logger.info("Pruning early stop")
                    break
            del self.pruner
            torch.cuda.empty_cache()
        plot_pruning_performance(self.sparsity_list, # x
                                 self.macs_list, # y1
                                 self.nparams_list, # y2
                                 self.pruned_map_list, # y3
                                 self.map_list, # y4,
                                 self.speedup_list,
                                 save_dir=self.args.output_dir)
        # self.model.export(format='onnx')


    def fine_tune(self, iter_step_idx):
        for name, param in self.model.model.named_parameters():
            param.requires_grad = True
        self.cfg.name = f"step_{iter_step_idx}_finetune"
        self.cfg.batch = self.batch_size  # restore batch size
        self.cfg.device = self.device
        self.model.train_v2(pruning=True, **vars(self.cfg))

        # post fine-tuning validation
        self.cfg.name = f"finetune_step_{iter_step_idx}_post_val"
        self.cfg.batch = self.batch_size
        validation_model = YOLO(self.model.trainer.best)
        metrics = validation_model.val(**vars(self.cfg))
        finetune_speed = metrics.speed['inference']
        current_speed_up = self.base_speed / finetune_speed
        self.speedup_list.append(current_speed_up)
        self.current_map = metrics.pose.map50 # different vs. utlis/eval.py/eval_engine
        self.logger.info(f"finetuned step {iter_step_idx} best model->->speed:{metrics.speed}, "
                         f"speedup:{current_speed_up}")

        self.logger.info(f"fintuned model->box map50-95:{metrics.box.map}, box map50:{metrics.box.map50}"
                         f"pose map50-95:{metrics.pose.map}, pose map50:{metrics.pose.map50}")

        self.logger.info(f"After fine tuning mAP={self.current_map}")

        self.map_list.append(self.current_map)
        # remove pruner after single iteration

        # save_pruning_performance_graph(self.nparams_list, self.map_list, self.macs_list, self.pruned_map_list)


    def build_engine(self, iter=None):
        def load_model(org_model_path, pt_model_path, device='cuda:0')-> PoseModel:
            model = YOLO(org_model_path)
            model1 = torch.load(pt_model_path, map_location=device)["model"]
            model1.float()
            model1.eval()
            with torch.no_grad():
                model1.fuse()
            model.model = model1
            model.args = vars(model.args)
            model.model.args = model.args
            model.model.task = model.task
            return model
        if iter is None:
            iter = self.iterative_steps-1 # final

        best_pt_path = os.path.join(self.args.output_dir, f"step_{iter}_finetune", "weights", "best.pt")
        pruned_model = load_model(self.weight, best_pt_path)
        # fp16 trt
        engine_path = pruned_model.export(format='engine', dynamic=self.args.dynamic, imgsz=self.args.imgsz,
                                          verbose=False, batch=self.args.batch_size, workspace=2, half=True)
        # export onnx, dynamic batch, shape fix
        print(engine_path)
        os.rename(engine_path, self.args.engine_path) # weights/yolov8-pose-prune.engine
        if os.path.exists(self.args.engine_path):
            self.logger.info(f"build engine success: {self.args.engine_path}")
            return True
        else:
            self.logger.info("build engine failed!")
            return False




