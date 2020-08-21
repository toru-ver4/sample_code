# -*- coding: utf-8 -*-
"""
Visualize the effect of the Crosstalk Matrix
============================================

1. design the tone mapping parameters to cover 10,000 cd/m2
2. do the simple tone mapping and comare the two color volumes,
   one is the sdr color volume and the other is sdr color volume
   that is converted by tone mapping.
3. do the tone mapping with crosstalk matrix
   and compare the two color volumes.

"""

# import standard libraries
import os

# import third-party libraries

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
