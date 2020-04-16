from qiskit import *
import numpy as np
import networkx as nx
import random as ran
import math
import cmath
from qiskit.visualization import *
from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus
import scipy as sp 
#import matplotlib as mpl
import sys
sys.path.append('../')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

print('importing done.')

def testu3(theta_c, phi_c, lamb_c, theta_x, phi_x, lamb_x, real, ideal):
    I = complex(0.0, 1.0)
    a_c = math.cos(theta_c/2)
    b_c = - cmath.exp(I*lamb_c)*math.sin(theta_c/2)
    c_c = cmath.exp(I*phi_c)*math.sin(theta_c/2)
    d_c = cmath.exp(I*(lamb_c + phi_c))*math.cos(theta_c/2)
    a_x = math.cos(theta_x/2)
    b_x = - cmath.exp(I*lamb_x)*math.sin(theta_x/2)
    c_x = cmath.exp(I*phi_x)*math.sin(theta_x/2)
    d_x = cmath.exp(I*(lamb_x + phi_x))*math.cos(theta_x/2)
    res00 = abs(a_c*a_x*real[0] + a_c*b_x*real[1] + b_c*a_x*real[2] + b_c*b_x*real[3])
    res01 = abs(a_c*c_x*real[0] + a_c*d_x*real[1] + b_c*c_x*real[2] + b_c*d_x*real[3])
    res10 = abs(c_c*a_x*real[0] + c_c*b_x*real[1] + d_c*a_x*real[2] + d_c*b_x*real[3])
    res11 = abs(c_c*c_x*real[0] + c_c*d_x*real[1] + d_c*c_x*real[2] + d_c*d_x*real[3])
    res = [res00, res01, res10, res11]
    d = sp.spatial.distance.euclidean(res,ideal)
    #print(real, '--> Uc(',theta_c, phi_c, lamb_c, '), Ux(',theta_x, phi_x, lamb_x, ') -->', res, d)
    #print(d)
    return d

def calcu3(theta_c, phi_c, lamb_c, theta_x, phi_x, lamb_x, real):
    I = complex(0.0, 1.0)
    a_c = math.cos(theta_c/2)
    b_c = - cmath.exp(I*lamb_c)*math.sin(theta_c/2)
    c_c = cmath.exp(I*phi_c)*math.sin(theta_c/2)
    d_c = cmath.exp(I*(lamb_c + phi_c))*math.cos(theta_c/2)
    a_x = math.cos(theta_x/2)
    b_x = - cmath.exp(I*lamb_x)*math.sin(theta_x/2)
    c_x = cmath.exp(I*phi_x)*math.sin(theta_x/2)
    d_x = cmath.exp(I*(lamb_x + phi_x))*math.cos(theta_x/2)
    res00 = abs(a_c*a_x*real[0] + a_c*b_x*real[1] + b_c*a_x*real[2] + b_c*b_x*real[3])
    res01 = abs(a_c*c_x*real[0] + a_c*d_x*real[1] + b_c*c_x*real[2] + b_c*d_x*real[3])
    res10 = abs(c_c*a_x*real[0] + c_c*b_x*real[1] + d_c*a_x*real[2] + d_c*b_x*real[3])
    res11 = abs(c_c*c_x*real[0] + c_c*d_x*real[1] + d_c*c_x*real[2] + d_c*d_x*real[3])
    res = [res00, res01, res10, res11]
    
    #print(real, '--> Uc(',theta_c, phi_c, lamb_c, '), Ux(',theta_x, phi_x, lamb_x, ') -->', res)
    return res

def approxu3(A, B):
    real = []
    ideal = []
    for i in range(4):
        real.append(math.sqrt(A[i]/sum(A)))
        ideal.append(math.sqrt(B[i]/sum(B)))
    #print(real, ideal)
    print(sp.spatial.distance.euclidean(A,B))
    theta_c = 0.0
    phi_c = 0.0
    lamb_c = 0.0
    theta_x = 0.0
    phi_x = 0.0
    lamb_x = 0.0
    d = testu3(theta_c, phi_c, lamb_c, theta_x, phi_x, lamb_x, real, ideal)
    for i in np.arange(1,30):
        if 5<i:
            k = (i-5)**2
        else:
            k = i
        d_new = testu3(theta_c+(1/k), phi_c, lamb_c, theta_x, phi_x, lamb_x, real, ideal)
        if d_new < d:
            theta_c = theta_c+(1/k)
            d = d_new
        else:
            d_new = testu3(theta_c-(1/k), phi_c, lamb_c, theta_x, phi_x, lamb_x, real, ideal)
            if d_new < d:
                theta_c = theta_c-(1/k)
                d = d_new

        d_new = testu3(theta_c, phi_c+(1/k), lamb_c, theta_x, phi_x, lamb_x, real, ideal)
        if d_new < d:
            phi_c = phi_c+(1/k)
            d = d_new
        else:
            d_new = testu3(theta_c, phi_c-(1/k), lamb_c, theta_x, phi_x, lamb_x, real, ideal)
            if d_new < d:
                phi_c = phi_c-(1/k)
                d = d_new

        d_new = testu3(theta_c, phi_c, lamb_c+(1/k), theta_x, phi_x, lamb_x, real, ideal)
        if d_new < d:
            lamb_c = lamb_c+(1/k)
            d = d_new
        else:
            d_new = testu3(theta_c, phi_c, lamb_c-(1/k), theta_x, phi_x, lamb_x, real, ideal)
            if d_new < d:
                lamb_c = lamb_c-(1/k)
                d = d_new

        d_new = testu3(theta_c, phi_c, lamb_c, theta_x+(1/k), phi_x, lamb_x, real, ideal)
        if d_new < d:
            theta_x = theta_x+(1/k)
            d = d_new
        else:
            d_new = testu3(theta_c, phi_c, lamb_c, theta_x-(1/k), phi_x, lamb_x, real, ideal)
            if d_new < d:
                theta_x = theta_x-(1/k)
                d = d_new

        d_new = testu3(theta_c, phi_c, lamb_c, theta_x, phi_x+(1/k), lamb_x, real, ideal)
        if d_new < d:
            phi_x = phi_x+(1/k)
            d = d_new
        else:
            d_new = testu3(theta_c, phi_c, lamb_c, theta_x, phi_x-(1/k), lamb_x, real, ideal)
            if d_new < d:
                phi_x = phi_x-(1/k)
                d = d_new

        d_new = testu3(theta_c, phi_c, lamb_c, theta_x, phi_x, lamb_x+(1/k), real, ideal)
        if d_new < d:
            lamb_x = lamb_x+(1/k)
            d = d_new
        else:
            d_new = testu3(theta_c, phi_c, lamb_c, theta_x, phi_x, lamb_x-(1/k), real, ideal)
            if d_new < d:
                lamb_x = lamb_x-(1/k)
                d = d_new
    res = calcu3(theta_c, phi_c, lamb_c, theta_x, phi_x, lamb_x, real)
    for k in range(4):
        res[k] *= res[k]
    print(real, '--> Uc(',theta_c, phi_c, lamb_c, '), Ux(',theta_x, phi_x, lamb_x, ') \n -->', res, d)    
    return [theta_c, phi_c, lamb_c], [theta_x, phi_x, lamb_x]


def apply_ideal_cx(statevec, c, x):
    res = statevec[:]
    cc = 2**c
    xx = 2**x
    control = 0
    for i in np.arange(0,len(statevec),cc):
        print('i = ',i)
        if control:
            control = 0
            for j in np.arange(i,i+cc):
                print('j = ',j, j-xx)
                res[j] = statevec[j-xx]
                res[j-xx] = statevec[j]
                print(res)
        else:
            control = 1
            
    return res

"""
A = [.5,.0,.2,.3]
B = [50,.0,.0,50]


print(approxu3(A,B))

#print(apply_ideal_cx(A,1,0))

"""