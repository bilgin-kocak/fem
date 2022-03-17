# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:31:07 2022

@author: kocak
"""
import numpy as np

def get_shape_func(x, y):
    assert len(x)==4, "The length of the x coordinate should be 4 for eact element."
    
    ksi_eta = [(-1,-1),
               ( 1,-1),
               ( 1, 1),
               (-1, 1)]
    
    Ns = list(map(lambda x: lambda ksi, eta: (1+ksi*x[0])*(1+eta*x[1])/4, ksi_eta))
    Ns_ksi = list(map(lambda x: lambda ksi, eta: ksi*(1+eta*x[1])/4, ksi_eta))
    Ns_eta = list(map(lambda x: lambda ksi, eta: (1+ksi*x[0])*eta/4, ksi_eta))
    # invJ = 
    
    
    return Ns, Ns_ksi, Ns_eta

    

def gaussian(n):
    
    if n==1:
        weights = [4]
        gauss_points = [(0,0)]
    elif n==4:
        weights = [1,1,1,1]
        gauss_points = [(-1/np.sqrt(3), -1/np.sqrt(3)), 
                        ( 1/np.sqrt(3), -1/np.sqrt(3)),
                        ( 1/np.sqrt(3),  1/np.sqrt(3)),
                        (-1/np.sqrt(3),  1/np.sqrt(3))]
    elif n==9:
        gps, ws = np.polynomial.legendre.leggauss()
        weights = []
        gauss_points = []
        for i in range(len(gps)):
            for j in range(len(gps)):
                weights.append(ws[i]*ws[j])
                gauss_points.append((gps[i], gps[j]))
                
        
    return weights, gauss_points



def calc_2dgauss_points(deg):
    gps, ws = np.polynomial.legendre.leggauss(deg)
    weights = []
    gauss_points = []
    for i in range(len(gps)):
        for j in range(len(gps)):
            weights.append(ws[i]*ws[j])
            gauss_points.append((gps[j], gps[i]))
                
    return gauss_points, weights