# -*- coding: utf-8 -*-
"""

===================================
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries


# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # tiff2png()

    max_luminance = 10000
    min_luminance = 10

    max_cv = np.round(np.log2(max_luminance / 50) * 32)
    print(max_cv)

    min_cv = np.round(((min_luminance / max_luminance * 100) ** 0.5) * 255)
    print(min_cv)

    print(50 * 2 ** (max_cv/32))
    print(max_luminance * ((min_cv / 255) ** 2) / 100)
