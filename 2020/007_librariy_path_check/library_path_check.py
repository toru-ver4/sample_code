# -*- coding: utf-8 -*-
"""
ライブラリのパスチェック
=======================
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import test_pattern_generator2 as tpg
import plot_utility as pu
import matplotlib.pyplot as plt
import transfer_functions as tf

# import my libraries
# import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(tpg.equal_devision(10, 3))
    x = np.linspace(0, 1, 1024)
    y = tf.eotf_to_luminance(x, tf.GAMMA24)
    ax1 = pu.plot_1_graph()
    ax1.plot(x, y)
    plt.show()
