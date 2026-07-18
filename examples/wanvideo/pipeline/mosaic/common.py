import os
import types
import glob as _glob
import json
import random
import re
import time
import contextlib
import html
import fcntl
from collections import defaultdict, deque
from datetime import datetime
from argparse import SUPPRESS

import accelerate
import argparse
import imageio
import numpy as np
import torch
import warnings
from PIL import Image, ImageDraw, ImageFont

from diffsynth.core import (
    DA3MosaicVideoDataset,
    SubjectRefMemoryDA3MosaicVideoDataset,
)
from diffsynth.core.data.unified_dataset import _derive_seed
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(
    "[PYBOOT] "
    f"host={os.uname().nodename} "
    f"RANK={os.environ.get('RANK')} "
    f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
    f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
    f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
    f"MASTER_PORT={os.environ.get('MASTER_PORT')}",
    flush=True,
)
