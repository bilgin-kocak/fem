# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 19:23:03 2022

@author: kocak
"""

import shape_func as sf
import numpy as np

gp, ws = sf.calc_2dgauss_points(2)

def phi(x, y):
    return x

x = [0,0,1,1]
y = [0,1,1,0]

# integral phi*dx*dy

phi_nodes = list(map(phi, x,y))

Ns, Ns_ksi, Ns_eta = sf.get_shape_func(x,y)

integral =  sum(list(map(lambda N, phi_n: N(gp[0][0], gp[0][1])*phi_n, Ns, phi_nodes)))


def calc_B(ksi, eta):
    return np.array([[1, 2],
                     [1, 2]])