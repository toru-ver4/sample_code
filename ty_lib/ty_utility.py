
# -*- coding: utf-8 -*-
"""
utility
"""

# import standard libraries
import os
from pathlib import Path

# import third-party libraries

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def add_suffix_to_filename(fname="./img/hoge.png", suffix="_with_icc"):
    """
    Parameters
    ----------
    suffix : str
        A suffix.
    fname : str
        A file name.

    Examples
    --------
    >>> add_suffix_to_filename(fname="./img/hoge.png", suffix="_with_icc")
    "img/hoge_with_icc.png"
    """
    pp = Path(fname)
    ext = pp.suffix
    parent = str(pp.parent)
    new_name = parent + "/" + pp.stem + suffix + ext

    return new_name


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fff = add_suffix_to_filename(suffix="_with_icc", fname="./img/hoge.png")
    print(fff)
