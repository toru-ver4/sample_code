# -*- coding: utf-8 -*-
"""
m4a ファイルのタグの編集
==============

"""

# import standard libraries
import os

# import third-party libraries
import mutagen

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def replace_name(filename):
    mo = mutagen.File(filename)
    name = mo.tags['\xa9nam'][0]
    name = name.replace('[TBSラジオ]', '')
    name = name.replace('（１） ', '_')
    name = name.replace('（２） ', '_')
    name = name.replace('(TimeFree)', '_')
    mo.tags['\xa9nam'][0] = name
    mo.save()


def main_func():
    dir = "./m4a_files"
    for filename in os.listdir(dir):
        replace_name(os.path.join(dir, filename))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()
