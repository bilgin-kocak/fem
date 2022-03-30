# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:13:31 2022

@author: kocak
"""

from fem import FEM
from meshr import read_mesh
import numpy as np
import matplotlib.pyplot as plt

def force(x, y):
    if fem.near(x, 0.0) and fem.near(y, 0.1):
        return (0, -100000/np.pi)
    else:
        return (0,0)
        
E = 9500e6
nu = 0.25
dim = 2  # dimension

fem = FEM(dim, E, nu, cons="axissymmetric")

mesh = read_mesh('ce522hw1q3.m')
p = mesh['POS']
t = mesh['QUADS'][:,:-1]
t = t -1

plt.figure()
fem.plot_mesh(p, t)

## Dirichlet Boundary Condition
keep_nodes = np.array([1,2])
keep_nodes_x = np.array([0,3])

ndof=dim*(p.shape[0])

K, F = fem.assemble_all(p, t, force)


fixed=np.concatenate((2*keep_nodes, 2*keep_nodes+1, 2*keep_nodes_x))
free = np.setdiff1d(np.arange(ndof), fixed)
values=np.zeros((len(fixed),));
F=F[free]-np.matmul(K[free,:][:,fixed], values);
K = K[free,:][:,free]
d=np.zeros((ndof,));
d[free] = np.linalg.solve(K, F)

U=d[::2]; V=d[1::2];
UU = U + p[:,0] 
VV = V + p[:,1]

plt.figure()
plt.scatter(UU, VV)
plt.title('Node Position After Loading')


def get_nodal_forces(node_number):
    element = np. where(t == node_number)
    nodes = t[element[0][0]]
    x=p[nodes,0]; y=p[nodes,1]; # node coordinates
    u= U[nodes]; v=V[nodes]; # nodal values
    dd = np.zeros((8,1)).squeeze()
    dd[0::2] = u;  dd[1::2] = v
    ke, me = fem.stiffness(x, y)

    fe = np.matmul(ke, dd)

    f_nodes = np.array([element[1][0]*2, element[1][0]*2+1])
    
    return fe[f_nodes]


def get_nodal_displacement(node_number):
    return U[node_number], V[node_number]
    


# Printing nodal forces at the dirichlet boundary nodes.
for node_number in keep_nodes:
    f1 = get_nodal_forces(node_number)
    print(f"For node {node_number}, fx={f1[0]} N, fy={f1[1]} N")

node_number = 0
ux, uy = get_nodal_displacement(node_number)
print(f"For node {node_number}, we have displacement ux={ux} m, uy={uy} m")
    
