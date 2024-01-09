# -*- coding: utf-8 -*-
"""
dump
"""

# import standard libraries
import os

# import third-party libraries
import PyOpenColorIO as ocio

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    builtin_transform_registry = ocio.BuiltinTransformRegistry()
    builtin_transforms = builtin_transform_registry.getBuiltins()
    print(type(builtin_transforms))
    for builtin_transform in builtin_transforms:
        name = builtin_transform[0]
        description = builtin_transform[1]
        print(f'name: "{name}", description: "{description}"')

    print(dir(ocio.BuiltinConfigRegistry))
