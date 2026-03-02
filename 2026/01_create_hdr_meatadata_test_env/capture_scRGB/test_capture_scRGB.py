from pathlib import Path
import os
import sys
import unittest

import numpy as np


class TestCapturescRGB(unittest.TestCase):
    def test_bitstream_metadata(self):
        pass

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    unittest.main()
