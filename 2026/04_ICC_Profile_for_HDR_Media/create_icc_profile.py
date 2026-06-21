# -*- coding: utf-8 -*-
"""
debug code
==========

"""

# import standard libraries
import os
import xml.etree.ElementTree as ET
import subprocess
from pathlib import Path
from itertools import product

# import third-party libraries
import numpy as np
from colour.utilities import tstack

# import my libraries
import icc_profile_xml_control as ipxc
import icc_profile_calc_param as ipcp
import color_space as cs
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    