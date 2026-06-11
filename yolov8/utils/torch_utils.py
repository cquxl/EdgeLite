from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms, nms

from .utils import obb_postprocess as np_obb_postprocess