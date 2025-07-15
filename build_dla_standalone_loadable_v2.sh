#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# TRTEXEC=/path/to/trtexec
# trtexec --onnx=ultralytics/weights/yolov8s-pose-qat.onnx --verbose --fp16 --saveEngine=ultralytics/weights.fp16chw16in.fp16chw16out.standalone.bin --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --buildDLAStandalone --useDLACore=0
# echo "Build DLA loadable for fp16 and int8"
# mkdir -p data/loadable
# trtexec --minShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --shapes=images:1x3x640x640 --onnx=ultralytics/weights/yolov8s-pose-qat.onnx --useDLACore=0 --buildDLAStandalone --saveEngine=ultralytics/weights/yolov8s_pose.int8.int8hwc4in.fp16chw16out.standalone.bin  --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8 --fp16 --calib=ultralytics/weights/yolov8s-pose-qat_precision_config_calib.cache --precisionConstraints=obey --layerPrecisions="/model.24/m.0/Conv":fp16,"/model.24/m.1/Conv":fp16,"/model.24/m.2/Conv":fp16,"/model.23/cv3/conv/Conv":fp16,"/model.23/cv3/act/Sigmoid":fp16,"/model.23/cv3/act/Mul":fp16
trtexec --onnx=ultralytics/weights/yolov8s-pose-qat.onnx \
        --saveEngine=ultralytics/weights/yolov8s-pose-qat.trt \
        --int8 
trtexec --onnx=weights/yolov8s-pose-qat.onnx --saveEngine=weights/yolov8s-pose-qat.engine --minShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --calib=./cache/trt_calibration.cache --int8 --useCudaGraph  # 启用CUDA图优化减少内核调度
trtexec --onnx=weights/yolov8s-pose-qat.onnx --saveEngine=weights/yolov8s-pose-qat.engine --minShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --int8 --fp16

trtexec --onnx=./weights/yolov8s-pose-qat-ops15.onnx --saveEngine=./weights/yolov8s-pose-qat-ops15.engine --minShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --int8 --fp16 --device=2 --calib=./cache/trt_calibration.cache

trtexec --minShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --shapes=images:1x3x640x640 --onnx=weights/yolov8s-pose-qat_noqdq.onnx --useDLACore=0 --buildDLAStandalone --saveEngine=weights/yolov8s-pose-qat_noqdq.engine  --inputIOFormats=int8:dla_hwc4 --int8 --fp16 --calib=./cache/trt_calibration.cache

trtexec --onnx=weights/yolov8s-pose-qat-cuda.onnx --saveEngine=weights/yolov8s-pose-qat-cuda.engine --minShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --calib=./cache/trt_calibration.cache --int8 

trtexec --onnx=weights/yolov8s-pose-qat-cuda_noqdq.onnx --saveEngine=weights/yolov8s-pose-qat-cuda_noqdq.engine --minShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --calib=./cache/trt_calibration.cache --int8 

trtexec --onnx=weights/yolov8s-pose-qat-cuda_noqdq.onnx --saveEngine=weights/yolov8s-pose-qat-cuda_noqdq.engine --minShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --calib=./cache/trt_calibration.cache --int8 

trtexec --onnx=weights/yolov8s-pose-qat-cuda-new_noqdq.onnx --saveEngine=weights/yolov8s-pose-qat-cuda-new_noqdq.engine --calib=./cache/trt_calibration.cache --int8 
trtexec --onnx=weights/yolov8s-pose-qat-cuda-new.onnx --saveEngine=weights/yolov8s-pose-qat-cuda-new.engine --calib=./cache/trt_calibration.cache --int8 

trtexec --onnx=weights/yolov8s-pose-qat-cuda-fixed.onnx --saveEngine=weights/yolov8s-pose-qat-cuda-fixed.engine --calib=./cache/trt_calibration.cache --int8 

# 静态
trtexec --onnx=yolov8s-pose.onnx --saveEngine=yolov8s-pose-qat.engine --shapes=images:1x3x640x640 --minShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --int8 --fp16 --verbose

trtexec --onnx=weights/yolov8s-pose-prune.onnx --saveEngine=weights/yolov8s-pose-prune-ptq.engine --shapes=images:1x3x640x640 --minShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --int8 --verbose --calib=cache/trt_calibration.cache

trtexec --device=0 --onnx=yolov8s-pose.onnx --saveEngine=yolov8s-pose-qat.engine --shapes=images:16x3x640x640 --minShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --int8 --fp16 --verbose
# 动态batch(需要确保导出onnx固定了imgsz)
trtexec --onnx=yolov8s-pose.onnx --saveEngine=yolov8s-pose-qat.engine --minShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --int8 --fp16 --verbose
# 动态batch与shape
trtexec --onnx=yolov8s-pose.onnx --saveEngine=yolov8s-pose-qat.engine --minShapes=images:1x3x320x320 --maxShapes=images:16x3x1280x1280 --optShapes=images:16x3x640x640 --int8 --fp16 --verbose

trtexec --onnx=output/yolov8s-pose-prune-sp0.5-epoch60/step_14_finetune/weights/best.onnx --saveEngine=weights/yolov8s-pose-prune-qat.engine --shapes=images:16x3x640x640 --minShapes=images:16x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --int8 --fp16 --verbose

trtexec --onnx=output/yolov8s-pose-prune-sp0.5-epoch60/step_14_finetune/weights/best.onnx --saveEngine=weights/yolov8s-pose-prune-qat.engine --minShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --int8 --fp16 --verbose --explicitBatch

trtexec --onnx=weights/yolov8s-pose-prune-qat.onnx --saveEngine=weights/yolov8s-pose-prune-qat1.engine --shapes=images:1x3x640x640 --minShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --int8 --fp16 --verbose

trtexec --onnx=weights/yolov8s-pose-prune-qat.onnx --saveEngine=weights/yolov8s-pose-prune-qat1.engine --shapes=images:1x3x640x640 --minShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --int8 --fp16 --verbose

trtexec --onnx=weights/yolov8s-pose-prune-qat.onnx --saveEngine=weights/yolov8s-pose-prune-qat1.engine --shapes=images:1x3x640x640 --minShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --int8 --verbose

trtexec --onnx=weights/yolov8s-pose-prune-qat1.onnx --saveEngine=weights/yolov8s-pose-prune-qat1.engine --int8 --verbose

trtexec --onnx=weights/yolov8s-pose.onnx --saveEngine=weights/yolov8s-pose-fp16.engine --minShapes=images:1x3x640x640 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --fp16 --verbose

trtexec --onnx=weights/yolov8s-pose.onnx --saveEngine=weights/yolov8s-pose-fp161.engine --minShapes=images:1x3x32x32 --maxShapes=images:16x3x640x640 --optShapes=images:16x3x640x640 --fp16 