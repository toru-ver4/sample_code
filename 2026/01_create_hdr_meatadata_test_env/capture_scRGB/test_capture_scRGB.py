import os
import unittest

import numpy as np
from imagecodecs import imread


class TestCapturescRGB(unittest.TestCase):
    def test_bitstream_metadata(self):
        ref_image = imread("./img/dst_windows_official_screenshot.jxr")
        test_image = imread("./img/dst_capture_scRGB_screenshot.jxr")
        np.testing.assert_almost_equal(test_image, ref_image)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    unittest.main()
