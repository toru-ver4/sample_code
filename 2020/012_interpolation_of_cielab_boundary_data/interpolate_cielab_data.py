# -*- coding: utf-8 -*-
"""
CIELAB 
==============

## 方針

HUE を求める。HUE を360 で割って正規化。
roundup(HUE * sample_num), rounddown(HUE * sample_num) で補間対象のindexが求まる？
L* 方向も同様で

"""

# import standard libraries
import os

# import third-party libraries

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
