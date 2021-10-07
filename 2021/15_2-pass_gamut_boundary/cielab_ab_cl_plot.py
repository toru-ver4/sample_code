# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np
from colour import Lab_to_XYZ, XYZ_to_RGB
from colour.models import RGB_COLOURSPACES
from cielab import is_inner_gamut

# import my libraries
import plot_utility as pu
import color_space as cs
import transfer_functions as tf
from create_gamut_booundary_lut import is_out_of_gamut_rgb
from jzazbz import jzazbz_to_large_xyz, jzczhz_to_jzazbz

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
