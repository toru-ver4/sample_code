from pathlib import Path
import os
import sys
import subprocess

import numpy as np
from OpenImageIO import (
    ImageSpec,
    ImageOutput,
    UINT16
)

import test_pattern_generator2 as tpg


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
