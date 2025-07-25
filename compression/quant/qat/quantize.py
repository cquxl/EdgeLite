from pytorch_quantization import nn as quant_nn
import torch
import torch.nn as nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from absl import logging as quant_logging
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization import quant_modules
from pytorch_quantization.nn.modules import _utils

from tqdm import tqdm
import torch.optim as optim
from torch.cuda import amp
from typing import List, Callable, Union, Dict
from copy import deepcopy
import re
import os

import onnx
from onnxsim import simplify

from .rules import find_quantizer_pairs


''' quantization method definition'''


class QuantConcat(torch.nn.Module, _utils.QuantInputMixin):
    def __init__(self, dimension=1):
        super(QuantConcat, self).__init__()
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
        self._input_quantizer._calibrator._torch_hist = True
        self.dimension = dimension

    def forward(self, inputs):
        inputs = [self._input_quantizer(input) for input in inputs]
        return torch.cat(inputs, self.dimension)


class QuantSiLU(torch.nn.Module, _utils.QuantInputMixin):
    def __init__(self, **kwargs):
        super(QuantSiLU, self).__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
        self._input0_quantizer._calibrator._torch_hist = True
        self._input1_quantizer._calibrator._torch_hist = True

    def forward(self, input):
        return self._input0_quantizer(input) * self._input1_quantizer(torch.sigmoid(input))


class QuantAdd(torch.nn.Module):
    def __init__(self, quantization):
        super().__init__()

        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input0_quantizer._calibrator._torch_hist = True
            self._fake_quant = True
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input0_quantizer(y)
        return x + y



class disable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True


def print_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(name, module)


# Initialize PyTorch Quantization
def initialize(all_node_with_qdq=False):
    quant_desc_input = QuantDescriptor(num_bits=8, calib_method="histogram")
    # quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_logging.set_verbosity(quant_logging.ERROR)
    os.environ["PTQ_MAX_AMAX"] = "100.0" # T4 add

    if all_node_with_qdq:
        quant_modules._DEFAULT_QUANT_MAP.extend(
            [quant_modules._quant_entry(torch.nn, "SiLU", QuantSiLU),
             quant_modules._quant_entry(models.common, "Concat", QuantConcat)]
        )


def transfer_torch_to_quantization(nninstance: torch.nn.Module, quantmodule, all_node_with_qdq):
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        if all_node_with_qdq and (self.__class__.__name__ == 'QuantSiLU' or self.__class__.__name__ == 'QuantConcat'):
            self.__init__()

        elif isinstance(self, quant_nn_utils.QuantInputMixin):
            quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def quantization_ignore_match(ignore_policy: Union[str, List[str], Callable], path: str) -> bool:
    if ignore_policy is None: return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):

        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy: return True
        for item in ignore_policy:
            if re.match(item, path):
                return True
    return False


def bottleneck_quant_forward(self, x):
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def replace_bottleneck_forward(model):
    for name, bottleneck in model.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            if bottleneck.add:
                if not hasattr(bottleneck, "addop"):
                    print(f"Add QuantAdd to {name}")
                    bottleneck.addop = QuantAdd(bottleneck.add)
                bottleneck.__class__.forward = bottleneck_quant_forward


def replace_to_quantization_module(model: torch.nn.Module, ignore_policy: Union[str, List[str], Callable] = None,
                                   all_node_with_qdq=False):
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                ignored = quantization_ignore_match(ignore_policy, path)
                if ignored:
                    print(f"Quantization: {path} has ignored.")
                    continue

                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id],
                                                                       all_node_with_qdq)
            if isinstance(submodule, torch.nn.SiLU) and "c2" in path:  # 例如，检测头中的SiLU
                print(f"对 {path} 使用更保守的量化")
                module._modules[name] = QuantSiLU(conservative=True)  # 自定义更稳定的QuantSiLU

    recursive_and_replace_module(model)


def get_attr_with_path(m, path):
    def sub_attr(m, names):
        name = names[0]
        value = getattr(m, name)

        if len(names) == 1:
            return value

        return sub_attr(value, names[1:])

    array = [item for item in re.split("\.|/", path) if item]
    return sub_attr(m, array)


def apply_custom_rules_to_quantizer(model: torch.nn.Module, export_onnx: Callable):
    # apply rules to graph
    export_onnx(model, "quantization-custom-rules-temp.onnx")
    pairs = find_quantizer_pairs("quantization-custom-rules-temp.onnx")
    print(pairs)
    for major, sub in pairs:
        print(f"Rules: {sub} match to {major}")
        get_attr_with_path(model, sub)._input_quantizer = get_attr_with_path(model, major)._input_quantizer
    os.remove("quantization-custom-rules-temp.onnx")

    for name, bottleneck in model.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            if bottleneck.add:
                print(f"Rules: {name}.add match to {name}.cv1")
                major = bottleneck.cv1.conv._input_quantizer
                bottleneck.addop._input0_quantizer = major
                bottleneck.addop._input1_quantizer = major


def calibrate_model(model: torch.nn.Module, dataloader, device, num_batch=25):
    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)

                    module._amax = module._amax.to(device)

    def collect_stats(model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                imgs = datas['img'].to(device, non_blocking=True).float() / 255
                model(imgs)

                if i >= num_batch:
                    break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    # model.model.fuse = False
    collect_stats(model, dataloader, device, num_batch=num_batch)
    compute_amax(model, method="mse")

def finetune(
        model: torch.nn.Module, train_dataloader, per_epoch_callback: Callable = None, preprocess: Callable = None,
        nepochs=20, early_exit_batchs_per_epoch=1000, lrschedule: Dict = None, fp16=True, learningrate=1e-5,
        supervision_policy: Callable = None
):
    origin_model = deepcopy(model).eval()
    disable_quantization(origin_model).apply()

    model.train()
    model.requires_grad_(True)

    scaler = amp.GradScaler(enabled=fp16)
    optimizer = optim.Adam(model.parameters(), learningrate)
    quant_lossfn = torch.nn.MSELoss()
    device = next(model.parameters()).device

    if lrschedule is None:
        lrschedule = {
            0: 1e-6,
            3: 1e-5,
            8: 1e-4,
            15: 1e-3,
        }

    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)

        return forward_hook

    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer): continue

        if supervision_policy:
            if not supervision_policy(mname, ml):
                continue

        supervision_module_pairs.append([ml, ori])

    for iepoch in range(nepochs):

        if iepoch in lrschedule:
            learningrate = lrschedule[iepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate

        model_outputs = []
        origin_outputs = []
        remove_handle = []

        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs)))
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        pbar = tqdm(train_dataloader, desc="QAT", total=early_exit_batchs_per_epoch)
        for ibatch, imgs in enumerate(pbar):

            if ibatch >= early_exit_batchs_per_epoch:
                break

            if preprocess:
                imgs = preprocess(imgs, device)

            # imgs = imgs.to(device)
            with amp.autocast(enabled=fp16):
                model(imgs)

                with torch.no_grad():
                    origin_model(imgs)

                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    quant_loss += quant_lossfn(mo, fo)

                model_outputs.clear()
                origin_outputs.clear()

            if fp16:
                # add
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                quant_loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(
                f"QAT Finetuning {iepoch + 1} / {nepochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        # You must remove hooks during onnx export or torch.save
        for rm in remove_handle:
            rm.remove()

        if iepoch % 5 == 0:
            if per_epoch_callback:
                if per_epoch_callback(model, iepoch, learningrate):
                    break


def export_onnx(model, args, file, **kwargs):
    """core export function: handle quantization nodes"""
    if hasattr(quant_nn, 'TensorQuantizer'):
        # save original settings
        original_amax_method = quant_nn.TensorQuantizer._get_amax
        # original_fake_quant_setting = quant_nn.TensorQuantizer.use_fb_fake_quant
        original_fake_quant_setting = quant_nn.TensorQuantizer.use_fake_quant
        original_forward = quant_nn.TensorQuantizer.forward
        # fix amax
        def _patched_get_amax(self, inputs=None):
            amax = self._amax.detach().clone()
            return torch.clamp(amax, min=1e-5)
        def patched_forward(self, inputs):
            """simple forward function for quantization nodes"""
            return inputs  # export quantization nodes
        # apply
        quant_nn.TensorQuantizer._get_amax = _patched_get_amax
        # quant_nn.TensorQuantizer.use_fb_fake_quant = True
        quant_nn.TensorQuantizer.use_fake_quant = True
        quant_nn.TensorQuantizer.forward = patched_forward
    try:
        torch.onnx.export(
            model,
            args,
            file,
            **kwargs,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
        )
    finally:
        # recover original settings
        if hasattr(quant_nn, 'TensorQuantizer'):
            quant_nn.TensorQuantizer._get_amax = original_amax_method
            # quant_nn.TensorQuantizer.use_fb_fake_quant = original_fake_quant_setting
            quant_nn.TensorQuantizer.use_fake_quant = original_fake_quant_setting
            quant_nn.TensorQuantizer.forward = original_forward
    # validate exported model
    validate_onnx(file)


def export_onnx1(model, input, file, *args, **kwargs):
    # quant_nn.TensorQuantizer.use_fb_fake_quant = True
    quant_nn.TensorQuantizer.use_fake_quant = True

    model.eval()
    with torch.no_grad():
        # torch.onnx.export(model, input, file, *args, **kwargs, do_constant_folding=True,)
        torch.onnx.export(model, input, file, *args, **kwargs)

    # quant_nn.TensorQuantizer.use_fb_fake_quant = False
    quant_nn.TensorQuantizer.use_fake_quant = False

def validate_onnx(file):
    model = onnx.load(file)
    model_simp, check = simplify(
        model,
        skip_fuse_bn=True,
        input_shapes={"images": [1, 3, 640, 640]},
        dynamic_input_shape=True
    )
    onnx.save(model_simp, file)
    print(f"onnx validate finishe , save to: {file}")
    print("========================\n")