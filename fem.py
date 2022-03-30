# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 19:05:32 2022

@author: kocak
"""

import numpy as np
import matplotlib.pyplot as plt


class FEM():
    
    def __init__(self, d, E, nu, t=1, gp_deg = 2, cons="plane_stress", dim=2):
        self.dim = dim
        self.cons = cons
        self.deg = gp_deg
        self.t = t
        self.d = d
        self.E = E
        self.nu = nu
        
        
    def get_shape_func(self):
        "The length of the x coordinate should be 4 for eact element."
        
        ksi_eta = [(-1,-1),
                   ( 1,-1),
                   ( 1, 1),
                   (-1, 1)]
        
        
        Ns = list(map(lambda x: lambda ksi, eta: (1+ksi*x[0])*(1+eta*x[1])/4, ksi_eta))
        Ns_ksi = list(map(lambda x: lambda ksi, eta: x[0]*(1+eta*x[1])/4, ksi_eta))
        Ns_eta = list(map(lambda x: lambda ksi, eta: (1+ksi*x[0])*x[1]/4, ksi_eta))    
        
        return Ns, Ns_ksi, Ns_eta
    
    def calc_2dgauss_points(self):
        gps, ws = np.polynomial.legendre.leggauss(self.deg)
        weights = []
        gauss_points = []
        for i in range(len(gps)):
            for j in range(len(gps)):
                weights.append(ws[i]*ws[j])
                gauss_points.append((gps[j], gps[i]))
                    
        return gauss_points, weights
        
    
    def stiffness(self, x, y):
        
        Ns, Ns_ksi, Ns_eta = self.get_shape_func()
        
        self.Ns = Ns
        self.Ns_ksi = Ns_ksi
        self.Ns_eta = Ns_eta
        
        gp, ws = self.calc_2dgauss_points()
        
        ke = sum(map(lambda gp, w : w*self.calc_stiffness_gp(x, y, gp[0], gp[1]), gp, ws))
        
        
        Me = sum(map(lambda gp, w: w*self.calc_M(x, y, gp[0], gp[1]), gp, ws))
        
        return ke, Me
        
        
    def calc_Nprime(self, k, e):
        """This function calculates N' matrix 2D 

        Args:
            k: ksi 
            e: eta.

        Returns:
            N' matrix

        """
        Ns_ksi = self.Ns_ksi
        Ns_eta = self.Ns_eta
        return np.array([[Ns_ksi[0](k,e), 0, Ns_ksi[1](k,e), 0, Ns_ksi[2](k,e), 0, Ns_ksi[3](k,e), 0],
                         [Ns_eta[0](k,e), 0, Ns_eta[1](k,e), 0, Ns_eta[2](k,e), 0, Ns_eta[3](k,e), 0],
                         [0, Ns_ksi[0](k,e), 0, Ns_ksi[1](k,e), 0, Ns_ksi[2](k,e), 0, Ns_ksi[3](k,e)],
                         [0, Ns_eta[0](k,e), 0, Ns_eta[1](k,e), 0, Ns_eta[2](k,e), 0, Ns_eta[3](k,e)]])
    
    def calc_J(self, x, y, ksi, eta ):
        """This function calculates N' matrix 2D 
        
        Args:
            x: x coordinates of element
            y: y coordinates of element
            ksi: ksi  value in gauss points
            eta: eta.
        
        Returns:
            Jacobian matrix
        """
        Ns_ksi = self.Ns_ksi
        Ns_eta = self.Ns_eta
        Ns_ksi_arr = np.array(list(map(lambda x: x(ksi, eta), Ns_ksi)))
        Ns_eta_arr = np.array(list(map(lambda x: x(ksi, eta), Ns_eta)))
        
        dx_dksi = np.dot(np.array(x), Ns_ksi_arr)
        dx_deta = np.dot(np.array(x), Ns_eta_arr)
        dy_dksi = np.dot(np.array(y), Ns_ksi_arr)
        dy_deta = np.dot(np.array(y), Ns_eta_arr)
        J =  np.array([[dx_dksi, dy_dksi],
                       [dx_deta, dy_deta]])
        return J
    
    def calc_N(self, ksi, eta):
        Ns = self.Ns
        
        Ns_arr = np.array(list(map(lambda x: x(ksi, eta), Ns))) 
        N = np.zeros((2,8))
        N[0,::2] = Ns_arr;    N[1,1::2] = Ns_arr
        return N
    
    def calc_M(self, x, y, ksi, eta):
        
        detJ = np.linalg.det(self.calc_J(x, y, ksi, eta))
        N = self.calc_N(ksi, eta)
        return np.matmul(N.T, N)*detJ
    
    def calc_LAMBDA(self, J):
        invJ = np.linalg.inv(J)
        return np.block([[invJ,               np.zeros((2, 2))],
                         [np.zeros((2, 2)),    invJ            ]])
    
    def calc_B(self, LAMBDA, Nprime, Ns_arr, x, y):
        T = np.array([[1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 1, 1, 0]])
        
        if self.cons == "axissymmetric":
            r = np.average(x)
            T = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 1, 1, 0],
                          [0, 0, 0, 0]])
            
        B = np.matmul(np.matmul(T, LAMBDA), Nprime)
        
        if self.cons == "axissymmetric":
            B[-1,:] = np.array([Ns_arr[0], 0, Ns_arr[1], 0, Ns_arr[2], 0, Ns_arr[3], 0])/r
        return B
    
    def calc_B_new(self, J, Ns_arr, Ns_ksi_arr, Ns_eta_arr, x,y):
        
        invJ = np.linalg.inv(J)
        dx_dksi = invJ
        pass
    
    def calc_D(self):
        E = self.E
        nu = self.nu
        # Plane stress
        if self.cons == "plane_stress":
            D = E/(1-nu**2)*np.array([[1,   nu,   0],
                                  [nu,  1,    0],
                                  [0,   0, (1-nu)/2]])
        elif self.cons == "plane_strain":
            D = E/(1+nu)/(1-2*nu)*np.array([[1-nu,   nu,         0],
                                           [nu,    1-nu,         0],
                                           [0,       0, (1-2*nu)/2]])
        elif self.cons == "axissymmetric":
            D = E/(1+nu)/(1-2*nu)*np.array([[1-nu,   nu,         0,   nu],
                                           [nu,    1-nu,         0,   nu],
                                           [0,       0, (1-2*nu)/2,    0],
                                           [nu,      nu,         0, 1-nu]])
        
        return D
    
    def calc_stiffness_gp(self, x, y, ksi, eta):
        
        Ns_arr = np.array(list(map(lambda x: x(ksi, eta), self.Ns))) 
        t = self.t
        Nprime = self.calc_Nprime(ksi, eta)
        
        J = self.calc_J(x, y, ksi, eta)
        LAMBDA = self.calc_LAMBDA(J)
        B = self.calc_B(LAMBDA, Nprime, Ns_arr, x, y)
        D = self.calc_D()
        
        if self.cons == "axissymmetric":
            r = np.average(x)
        else:
            r = 1
        return np.matmul(np.matmul(B.T, D), B)*t*np.linalg.det(J)*r
    
    def assemble_all(self,p,t, force):
        ndof=self.dim*(p.shape[0])

        K=np.zeros((ndof,ndof)) # allocate stiffness matrix
        M=np.zeros((ndof,ndof)) # allocate mass matrix
        F=np.zeros((ndof,1)).squeeze() # allocate load vector
        dofs = np.zeros((8,1)).squeeze().astype(int)
        for i in range(t.shape[0]):
            nodes=t[i,:4] # element nodes
            x=p[nodes,0]; y=p[nodes,1]; # node coordinates
            dofs[1::2]=2*nodes + 1
            dofs[0::2]=2*nodes
            f = [force(x[i],y[i]) for i in range(len(x))]
            fK=np.array([f[0][0], f[0][1], f[1][0], f[1][1],
                f[2][0], f[2][1], f[3][0], f[3][1]])
            ke, me = self.stiffness(x, y)
            # if ke.sum()<0:
            #     print(f"neg {i}")
            
            
            # dofs.sort()
            
            # for j in range(len(dofs)):
            #     for k in range(len(dofs)):
            #         K[dofs[j],dofs[k]] += ke[j,k] 
            for j in range(len(dofs)):
                K[dofs[j],dofs]=K[dofs[j],dofs]+ke[j];
            # K[dofs,:][:,dofs]=K[dofs,:][:,dofs]+ke;
            FK = np.matmul(me, fK)
            F[dofs]=F[dofs]+fK   
            
        return K, F

    
    @staticmethod
    def near(x, val, tol=1e-4):
        if np.abs(x - val) < tol:
            return True
        else:
            return False
    @staticmethod
    def plot_mesh(p,t):
        for i in range(len(p)):
            plt.text(p[i,0], p[i,1], f'{i}')
        for quad in t:
            x = [p[quad[0],0], p[quad[1],0], p[quad[2],0], p[quad[3],0], p[quad[0],0]]
            y = [p[quad[0],1], p[quad[1],1], p[quad[2],1], p[quad[3],1], p[quad[0],1]]
            plt.plot(x, y, 'k')
        plt.axis('equal')
        plt.title('mesh')
        plt.show()
            
