# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np

# import my libraries
from ty_display_pro_hl import read_xyz_and_save_to_csv_file

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def measure_func(output_fname):
    read_xyz_and_save_to_csv_file(result_fname=output_fname, ccss_file=None)


def main(output_fname: str):
    while True:
        # Wait for key input from the user
        print("Press Enter to start measurement.", end="")
        print("Press any other key to exit the program.")
        key_input = input("Enter a key: ")

        # If only Enter is pressed, call the function
        if key_input == '':
            measure_func(output_fname)
        else:
            print("Exiting the program.")
            break  # Exit the loop


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(output_fname="./mesured_luminance/iPhone_Photos.csv")
