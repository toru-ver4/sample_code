#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
トーンカーブ実装
"""

import os
import numpy as np
import plot_utility as pu
import matplotlib.pyplot as plt


M = np.array([[0.5, -1.0, 0.5],
              [-1.0, 1.0, 0.5],
              [0.5, 0.0, 0.0]])


RRT_PARAMS = {
    'coefsLow': [-4.0000000000, -4.0000000000, -3.1573765773, -0.4852499958,
                 1.8477324706, 1.8477324706],
    'coefsHigh': [-0.7185482425, 2.0810307172, 3.6681241237, 4.0000000000,
                  4.0000000000, 4.0000000000],
    'minPoint': [0.18 * (2 ** -15), 0.0001],
    'midPoint': [0.18, 4.8],
    'maxPoint': [0.18 * (2 ** 18), 10000.0],
    'slopeLow': 0.0,
    'slopeHigh': 0.0
}

HALF_MIN = np.finfo('float16').tiny


def rrt_tonecurve(x, param=RRT_PARAMS, plot=False):
    N_KNOTS_LOW = 4
    N_KNOTS_HIGH = 4
    logx = np.log10(np.fmax(x, HALF_MIN))

    if logx <= np.log10(param['minPoint'][0]):
        logy = logx * param['slopeLow']\
            + (np.log10(param['minPoint'][1]) - param['slopeLow'] * np.log10(param['minPoint'][0]))
    elif (logx > np.log10(param['minPoint'][0])) and (logx < np.log10(param['midPoint'][0])):
        knot_coord = (N_KNOTS_LOW - 1)\
            * (np.log(x) - np.log10(param['minPoint'][0]))\
            / (np.log10(param['midPoint'][0]) - np.log10(param['minPoint'][0]))
        j = np.int32(np.floor(knot_coord))
        t = knot_coord - j
        cf = np.array([param['coefsLow'][j], param['coefsLow'][j+1],
                       param['coefsLow'][j+2]])
        monomials = np.array([t * t, t, 1.0])
        logy = 
    elif (logx >= np.log10(param['midPoint'][0])) and (logx < np.log10(param['maxPoint'][0])):
        pass
    else:
        pass

    return 


ODT_PARAM_1000nits = {
    'coefsLow': [-4.9706219331, -3.0293780669, -2.1262, -1.5105, -1.0578,
                 -0.4668, 0.11938, 0.7088134201, 1.2911865799, 1.2911865799],
    'coefsHigh': [0.8089132070, 1.1910867930, 1.5683, 1.9483, 2.3083, 2.6384,
                  2.8595, 2.9872608805, 3.0127391195, 3.0127391195],
    'minPoint': [rrt_tonecurve(0.18 * (2 ** -12.), False), 0.0001],
    'midPoint': [rrt_tonecurve(0.18, False), 10.0],
    'maxPoint': [rrt_tonecurve(0.18 * (2 ** 10), False), 1000.0],
    'slopeLow': 3.0,
    'slopeHigh': 0.06
}


def odt_tonecurve(x, spline_param, plot=False):
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    x = np.linspace(0, 100, 1024)
    odt_tonecurve(x, x, plot=True)
    print(HALF_MIN)
