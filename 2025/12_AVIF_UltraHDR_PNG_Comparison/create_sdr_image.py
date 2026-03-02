import os

import numpy as np
from colour.io import write_image


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    width = 1920
    height = 1080
    sdr_img = np.ones((height, width, 3)) * 0.0
    write_image(sdr_img, "./src_img/ultrahdr_sdr_image.png")
