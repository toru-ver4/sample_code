# -*- coding: utf-8 -*-

# import standard libraries
import time
from multiprocessing import Pool, cpu_count, Array, Process

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


WIDTH = int(1920 * 4)
HEIGHT = int(1080 * 4)
COLOR_NUM = 3
PROCESS_NUM = 8


class MeasureExecTime():
    def __init__(self):
        self.clear_buf()

    def clear_buf(self):
        self.st_time = 0.0
        self.lap_st = 0.0
        self.ed_time = 00

    def start(self):
        self.st_time = time.time()
        self.lap_st = self.st_time

    def lap(self, msg="", rate=1.0):
        current = time.time()
        print(f"{msg}, lap time = {(current - self.lap_st)*rate:.5f} [sec]")
        self.lap_st = current

    def end(self):
        current = time.time()
        print(f"total time = {current - self.st_time:.5f} [sec]")
        self.clear_buf()


def thread_wrapper_apply_matrix(args):
    apply_matrix(**args)


def apply_matrix(idx: int, seed: int):
    met = MeasureExecTime()
    # sleep_coef = 0.2
    met.start()
    np.random.seed(seed)
    r_in = np.random.rand(HEIGHT, WIDTH)
    g_in = np.random.rand(HEIGHT, WIDTH)
    b_in = np.random.rand(HEIGHT, WIDTH)

    mtx = np.random.rand(3, 3)
    met.lap(msg=f"idx={idx}, A")

    r_out = mtx[0, 0] * r_in + mtx[0, 1] * g_in + mtx[0, 2] * b_in
    met.lap(msg=f"idx={idx}, B")

    g_out = mtx[1, 0] * r_in + mtx[1, 1] * g_in + mtx[1, 2] * b_in
    met.lap(msg=f"idx={idx}, C")

    b_out = mtx[2, 0] * r_in + mtx[2, 1] * g_in + mtx[2, 2] * b_in
    met.lap(msg=f"idx={idx}, D")

    img = np.dstack([r_out, g_out, b_out]).reshape(HEIGHT, WIDTH, COLOR_NUM)
    met.lap(msg=f"idx={idx}, E")

    return idx


def main_func_normal_loop():
    met = MeasureExecTime()
    met.start()
    process_num = PROCESS_NUM
    for idx in range(process_num):
        d = dict(idx=idx, seed=idx)
        apply_matrix(**d)

    met.end()


def main_func_multiprocess():
    met = MeasureExecTime()
    met.start()
    process_num = PROCESS_NUM
    args = []
    for idx in range(process_num):
        d = dict(idx=idx, seed=idx)
        args.append(d)

    with Pool() as pool:
        pool.map(thread_wrapper_apply_matrix, args)

    met.end()


if __name__ == '__main__':
    main_func_normal_loop()
    print("\n\n")
    main_func_multiprocess()
    # np.show_config()
