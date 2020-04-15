from qiskit import *
import numpy as np
#import networkx as nx
#import random as ran
import math
import cmath
import re 
#from qiskit.visualization import *
#from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus
#import scipy as sp 
import matplotlib as mpl
#import sys
#sys.path.append('../')
from GenerateRandomCircuit import *

def u3tomatrix(theta, phi, lamb):
    I = complex(0,1)
    m =[[cmath.cos(theta/2), -cmath.exp(I*lamb)*cmath.sin(theta/2)],\
        [cmath.exp(I*phi)*cmath.sin(theta/2), cmath.exp(I*(phi+lamb))*cmath.cos(theta/2)]]
    return m

def approxantiu3(theta, phi, lamb):
    theta2 = theta + 0
    phi2 = phi + 0
    lamb2 = lamb + 0
    m = u3tomatrix(theta, phi, lamb)
    for l in np.arange(1,50000):
        k = .1*l#**2
        m1 = u3tomatrix(theta2, phi2, lamb2)
        d1 = abs((m[0][0] - m1[1][1])**2 + (m[0][1] - m1[1][0])**2 + (m[1][0] - m1[0][1])**2 + (m[1][1] - m1[0][0])**2)
        #print(d1)

        m2 = u3tomatrix(theta2 + 1/k, phi2, lamb2)
        d2 = abs((m[0][0] - m2[1][1])**2 + (m[0][1] - m2[1][0])**2 + (m[1][0] - m2[0][1])**2 + (m[1][1] - m2[0][0])**2)
        if d2<d1:
            theta2 += 1/k
            d1 = d2
        m2 = u3tomatrix(theta2 - 1/k, phi2, lamb2)
        d2 = abs((m[0][0] - m2[1][1])**2 + (m[0][1] - m2[1][0])**2 + (m[1][0] - m2[0][1])**2 + (m[1][1] - m2[0][0])**2)
        if d2<d1:
            theta2 -= 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2 + 1/k, lamb2)
        d2 = abs((m[0][0] - m2[1][1])**2 + (m[0][1] - m2[1][0])**2 + (m[1][0] - m2[0][1])**2 + (m[1][1] - m2[0][0])**2)
        if d2<d1:
            phi2 += 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2 - 1/k, lamb2)
        d2 = abs((m[0][0] - m2[1][1])**2 + (m[0][1] - m2[1][0])**2 + (m[1][0] - m2[0][1])**2 + (m[1][1] - m2[0][0])**2)
        if d2<d1:
            phi2 -= 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2, lamb2 + 1/k)
        d2 = abs((m[0][0] - m2[1][1])**2 + (m[0][1] - m2[1][0])**2 + (m[1][0] - m2[0][1])**2 + (m[1][1] - m2[0][0])**2)
        if d2<d1:
            lamb2 += 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2, lamb2- 1/k)
        d2 = abs((m[0][0] - m2[1][1])**2 + (m[0][1] - m2[1][0])**2 + (m[1][0] - m2[0][1])**2 + (m[1][1] - m2[0][0])**2)
        if d2<d1:
            lamb2 -= 1/k
            d1 = d2
        if 4*math.pi < theta2:
            theta2 -= 4*math.pi
        if 2*math.pi < phi2:
            phi2 -= 2*math.pi
        if 2*math.pi < lamb2:
            lamb2 -= 2*math.pi
        if theta2 < 0:
            theta2 += 4*math.pi
        if phi2 < 0:
            phi2 += 2*math.pi
        if lamb2 < 0:
            lamb2 += 2*math.pi
        if d1 < 0.00000000001:
            break
    return theta2, phi2, lamb2, d1

def approx3rdrootabs(theta, phi, lamb):
    theta2 = theta/3
    phi2 = phi/3
    lamb2 = lamb/3
    theta2 = 8
    phi2 = 5
    lamb2 = 1
    m = u3tomatrix(theta, phi, lamb)
    
    for l in np.arange(2,500):
        k = math.log(l)/10 #**2
        #k = .1*l
        u = u3tomatrix(theta2, phi2, lamb2)
        m1 = np.matmul(u,u)
        m1 = np.matmul(u,m1)
        d1 = abs(abs(m[0][0]) - abs(m1[0][0]))**2 + abs(m[0][1] - m1[0][1])**2 + abs(m[1][0] - m1[1][0])**2 + abs(abs(m[1][1]) - abs(m1[1][1]))**2
        #print(d1)

        m2 = u3tomatrix(theta2 + 1/k, phi2, lamb2)
        d2 = abs(abs(m[0][0]) - abs(m2[0][0]))**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(abs(m[1][1]) - abs(m2[1][1]))**2
        if d2<d1:
            theta2 += 1/k
            d1 = d2
        m2 = u3tomatrix(theta2 - 1/k, phi2, lamb2)
        d2 = abs(abs(m[0][0]) - abs(m2[0][0]))**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(abs(m[1][1]) - abs(m2[1][1]))**2
        if d2<d1:
            theta2 -= 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2 + 1/k, lamb2)
        d2 = abs(abs(m[0][0]) - abs(m2[0][0]))**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(abs(m[1][1]) - abs(m2[1][1]))**2
        if d2<d1:
            phi2 += 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2 - 1/k, lamb2)
        d2 = abs(abs(m[0][0]) - abs(m2[0][0]))**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(abs(m[1][1]) - abs(m2[1][1]))**2
        if d2<d1:
            phi2 -= 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2, lamb2 + 1/k)
        d2 = abs(abs(m[0][0]) - abs(m2[0][0]))**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(abs(m[1][1]) - abs(m2[1][1]))**2
        if d2<d1:
            lamb2 += 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2, lamb2- 1/k)
        d2 = abs(abs(m[0][0]) - abs(m2[0][0]))**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(abs(m[1][1]) - abs(m2[1][1]))**2
        if d2<d1:
            lamb2 -= 1/k
            d1 = d2
        if 4*math.pi < theta2:
            theta2 -= 4*math.pi
        if 2*math.pi < phi2:
            phi2 -= 2*math.pi
        if 2*math.pi < lamb2:
            lamb2 -= 2*math.pi
        if theta2 < 0:
            theta2 += 4*math.pi
        if phi2 < 0:
            phi2 += 2*math.pi
        if lamb2 < 0:
            lamb2 += 2*math.pi
        if d1 < 0.000001:
            break
        print(d1)
    return theta2, phi2, lamb2, d1

def approx3rdroot(theta, phi, lamb):
    theta2 = theta/3
    phi2 = phi/3
    lamb2 = lamb/3
    theta2 = 8
    phi2 = 5
    lamb2 = 1
    m = u3tomatrix(theta, phi, lamb)

    for l in np.arange(500,5000):
        #k = math.log(l)/10 #**2
        
        u = u3tomatrix(theta2, phi2, lamb2)
        m1 = np.matmul(u,u)
        m1 = np.matmul(u,m1)
        d1 = abs(m[0][0] - m1[0][0])**2 + abs(m[0][1] - m1[0][1])**2 + abs(m[1][0] - m1[1][0])**2 + abs(m[1][1] - m1[1][1])**2
        #print(d1)
        d_old = d1
        k = 10/d1
        m2 = u3tomatrix(theta2 + 1/k, phi2, lamb2)
        d2 = abs(m[0][0] - m2[0][0])**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(m[1][1] - m2[1][1])**2
        if d2<d1:
            theta2 += 1/k
            d1 = d2
        m2 = u3tomatrix(theta2 - 1/k, phi2, lamb2)
        d2 = abs(m[0][0] - m2[0][0])**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(m[1][1] - m2[1][1])**2
        if d2<d1:
            theta2 -= 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2 + 1/k, lamb2)
        d2 = abs(m[0][0] - m2[0][0])**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(m[1][1] - m2[1][1])**2
        if d2<d1:
            phi2 += 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2 - 1/k, lamb2)
        d2 = abs(m[0][0] - m2[0][0])**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(m[1][1] - m2[1][1])**2
        if d2<d1:
            phi2 -= 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2, lamb2 + 1/k)
        d2 = abs(m[0][0] - m2[0][0])**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(m[1][1] - m2[1][1])**2
        if d2<d1:
            lamb2 += 1/k
            d1 = d2
        m2 = u3tomatrix(theta2, phi2, lamb2- 1/k)
        d2 = abs(m[0][0] - m2[0][0])**2 + abs(m[0][1] - m2[0][1])**2 + abs(m[1][0] - m2[1][0])**2 + abs(m[1][1] - m2[1][1])**2
        if d2<d1:
            lamb2 -= 1/k
            d1 = d2
        
        if 4*math.pi < theta2:
            theta2 -= 4*math.pi
        if 2*math.pi < phi2:
            phi2 -= 2*math.pi
        if 2*math.pi < lamb2:
            lamb2 -= 2*math.pi
        if theta2 < 0:
            theta2 += 4*math.pi
        if phi2 < 0:
            phi2 += 2*math.pi
        if lamb2 < 0:
            lamb2 += 2*math.pi
        if d1 < 0.000001:
            break
        if (d1 - d_old <0.0001 and 0.01 < d1):
            theta2 = np.random.random_sample() * math.pi * 4
            phi2 = np.random.random_sample() * math.pi * 2
            lamb2 = np.random.random_sample() * math.pi * 2
            l = 500
            print(theta2, phi2, lamb2)
        print(l, d1, d2, d_old)
    return theta2, phi2, lamb2, d1


def internal_circuit_from_qasm(qasm):
    circ = []
    num_qbits = 0
    num_cbits = 0
    for line in iter(qasm.splitlines()):
        if line.startswith('qreg'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    num_qbits = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
        elif line.startswith('creg'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    num_cbits = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
        elif line.startswith('barrier'):
            circ.append(['barrier'])
        elif line.startswith('cx'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    c = int(item.group(0).lstrip('[').rstrip(']'))
                elif count==1:
                    x = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
            circ.append(['cx', c, x])
        elif line.startswith('u3'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    q = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
            search_results = re.finditer(r'\(.*?\)', line)
            count=0
            for item in search_results:
                if count==0:
                    tpl = item.group(0).lstrip('(').rstrip(')').split(',')
                    theta = float(tpl[0])
                    phi = float(tpl[1])
                    lamb = float(tpl[2])
                count+=1
            circ.append(['u3', theta, phi, lamb, q])
        elif line.startswith('u2'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    q = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
            search_results = re.finditer(r'\(.*?\)', line)
            count=0
            for item in search_results:
                if count==0:
                    tpl = item.group(0).lstrip('(').rstrip(')').split(',')
                    theta = 0
                    phi = float(tpl[0])
                    lamb = float(tpl[1])
                count+=1
            circ.append(['u3', theta, phi, lamb, q])
        elif line.startswith('u1'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    q = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
            search_results = re.finditer(r'\(.*?\)', line)
            count=0
            for item in search_results:
                if count==0:
                    tpl = item.group(0).lstrip('(').rstrip(')').split(',')
                    theta = 0
                    phi = 0
                    lamb = float(tpl[0])
                count+=1
            circ.append(['u3', theta, phi, lamb, q])
        elif line.startswith('measure'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    q = int(item.group(0).lstrip('[').rstrip(']'))
                elif count==1:
                    c = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
            circ.append(['measure', q, c])
        elif line.startswith('x'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    q = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
            circ.append(['x', q])
        elif line.startswith('y'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    q = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
            circ.append(['u3', math.pi, math.pi/2, math.pi/2, q])
        elif line.startswith('z'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    q = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
            circ.append(['u3', 0, math.pi, 0, q])
        elif line.startswith('h'):
            search_results = re.finditer(r'\[.*?\]', line)
            count=0
            for item in search_results:
                if count==0:
                    q = int(item.group(0).lstrip('[').rstrip(']'))
                count+=1
            circ.append(['u3', math.pi/2, 0, math.pi, q])
    #print(circ)
    return circ, num_qbits, num_cbits

def anticircuit(circ, num_qbits, num_cbits):
    anti = circ[:]
    d = 0
    while d < len(anti):
        k = anti[d][:]
        if k[0] == 'cx':
            anti.insert(d, ['x', k[1]])
            d += 2
            anti.insert(d, ['x', k[1]])
            d -=1
        elif k[0] == 'u3':
            theta2, phi2, lamb2, dummy = approxantiu3(k[1], k[2], k[3])
            k[1] = theta2
            k[2] = phi2
            k[3] = lamb2
            anti[d] = k
        elif k[0] == 'x':
            dummy = 'x'
        d += 1

    k = ['measure']
    while k[0] == 'measure':
        d -= 1
        k = anti[d][:]
    d+=1
    #anti.insert(d, ['barrier'])
    for i in range(num_qbits):
        anti.insert(d, ['x', i])
    anti.insert(d, ['barrier'])
    for i in range(num_qbits):
        anti.insert(0, ['x', i])
    #anti.append(['flipcbits'])
    #print(circ)
    return anti, num_qbits, num_cbits

def flip_at_start(circ, num_qbits, num_cbits):
    anti = circ[:]
    d = 0
    while d < len(anti):
        k = anti[d][:]
        if k[0] == 'cx':
            anti.insert(d, ['x', k[1]])
            d += 2
            anti.insert(d, ['x', k[1]])
            d -=1
        elif k[0] == 'u3':
            theta2, phi2, lamb2, dummy = approxantiu3(k[1], k[2], k[3])
            k[1] = theta2
            k[2] = phi2
            k[3] = lamb2
            anti[d] = k
        elif k[0] == 'x':
            dummy = 'x'
        d += 1

    for i in range(num_qbits):
        anti.insert(0, ['x', i])
    #anti.append(['flipcbits'])
    #print(circ)
    return anti, num_qbits, num_cbits

def flip_at_end(circ, num_qbits, num_cbits):
    anti = circ[:]
    
    d = len(anti)
    k = ['measure']
    while k[0] == 'measure':
        d -= 1
        k = anti[d][:]
    d+=1
    #anti.insert(d, ['barrier'])
    for i in range(num_qbits):
        anti.insert(d, ['x', i])
    anti.insert(d, ['barrier'])
    return anti, num_qbits, num_cbits

def times3withoutbarriers(circ, num_qbits, num_cbits):
    times3 = circ[:]
    d = 0
    while d < len(times3):
        k = times3[d][:]
        if k[0] == 'cx':
            times3.insert(d, k)
            times3.insert(d, k)
            d+= 2
        elif k[0] == 'u3':
            times3.insert(d, ['u3',math.pi, 0, math.pi, k[4]])
            times3.insert(d, ['u3',math.pi, 0, math.pi, k[4]])
            d += 2
        elif k[0] == 'x':
            times3.insert(d, k)
            times3.insert(d, k)
            d+= 2
        d += 1

    #print(times3)
    return times3, num_qbits, num_cbits


def times3(circ, num_qbits, num_cbits):
    times3 = circ[:]
    d = 0
    while d < len(times3):
        k = times3[d][:]
        if k[0] == 'cx':
            times3.insert(d,['barrier'])
            times3.insert(d, k)
            times3.insert(d,['barrier'])
            times3.insert(d, k)
            d += 4
            times3.insert(d,['barrier'])
            d+= 1
        elif k[0] == 'u3':
            times3.insert(d,['barrier'])
            times3.insert(d, ['u3',math.pi, 0, math.pi, k[4]])
            times3.insert(d,['barrier'])
            times3.insert(d, ['u3',math.pi, 0, math.pi, k[4]])
            d += 4
            times3.insert(d,['barrier'])
            d+= 1
        elif k[0] == 'x':
            times3.insert(d,['barrier'])
            times3.insert(d, k)
            times3.insert(d,['barrier'])
            times3.insert(d, k)
            d += 4
            times3.insert(d,['barrier'])
            d+= 1
        d += 1

    #print(times3)
    return times3, num_qbits, num_cbits

def times5(circ, num_qbits, num_cbits):
    times5 = circ[:]
    d = 0
    while d < len(times5):
        k = times5[d][:]
        if k[0] == 'cx':
            times5.insert(d,['barrier'])
            times5.insert(d, k)
            times5.insert(d,['barrier'])
            times5.insert(d, k)
            times5.insert(d,['barrier'])
            times5.insert(d, k)
            times5.insert(d,['barrier'])
            times5.insert(d, k)
            d += 8
            times5.insert(d,['barrier'])
            d+= 1
        elif k[0] == 'u3':
            times5.insert(d,['barrier'])
            times5.insert(d, ['u3',math.pi, 0, math.pi, k[4]])
            times5.insert(d,['barrier'])
            times5.insert(d, ['u3',math.pi, 0, math.pi, k[4]])
            times5.insert(d,['barrier'])
            times5.insert(d, ['u3',math.pi, 0, math.pi, k[4]])
            times5.insert(d,['barrier'])
            times5.insert(d, ['u3',math.pi, 0, math.pi, k[4]])
            d += 8
            times5.insert(d,['barrier'])
            d+= 1
        elif k[0] == 'x':
            times5.insert(d,['barrier'])
            times5.insert(d, k)
            times5.insert(d,['barrier'])
            times5.insert(d, k)
            times5.insert(d,['barrier'])
            times5.insert(d, k)
            times5.insert(d,['barrier'])
            times5.insert(d, k)
            d += 8
            times5.insert(d,['barrier'])
            d+= 1
        d += 1

    return times5, num_qbits, num_cbits


def circuit_from_internal(internalcirc, num_qbits, num_cbits):
    q = QuantumRegister(num_qbits)
    c = ClassicalRegister(num_cbits)
    circ = QuantumCircuit(q,c)
    for k in internalcirc:
        #print(k)
        if k[0] == 'x':
            circ.x(k[1])
        elif k[0] == 'cx':
            circ.cx(k[1], k[2])
        elif k[0] == 'u3':
            circ.u3(k[1], k[2], k[3], k[4])
        elif k[0] == 'measure':
            circ.measure(k[1], k[2])
        elif k[0] == 'flipcbits':
            print('You have to flip all classical bits!')
        elif k[0] == 'barrier':
            circ.barrier()
    return circ

def make_anti_circuit(circ):
    #print('qasm')
    qasm = circ.qasm()
    #print('internal')
    internal, num_qbits, num_cbits = internal_circuit_from_qasm(qasm)
    #print('anti')
    anti, num_qbits, num_cbits = anticircuit(internal, num_qbits, num_cbits)
    #print('circuit')
    anti_circuit = circuit_from_internal(anti, num_qbits, num_cbits)
    #print('done')
    return anti_circuit


def make_flip_at_start(circ):
    qasm = circ.qasm()
    internal, num_qbits, num_cbits = internal_circuit_from_qasm(qasm)
    flipped, num_qbits, num_cbits = flip_at_start(internal, num_qbits, num_cbits)
    flipped_circuit = circuit_from_internal(flipped, num_qbits, num_cbits)
    return flipped_circuit

def make_flip_at_end(circ):
    qasm = circ.qasm()
    internal, num_qbits, num_cbits = internal_circuit_from_qasm(qasm)
    flipped, num_qbits, num_cbits = flip_at_end(internal, num_qbits, num_cbits)
    flipped_circuit = circuit_from_internal(flipped, num_qbits, num_cbits)
    return flipped_circuit

def make_times3withoutbarriers(circ):
    qasm = circ.qasm()
    internal, num_qbits, num_cbits = internal_circuit_from_qasm(qasm)
    flipped, num_qbits, num_cbits = times3withoutbarriers(internal, num_qbits, num_cbits)
    circuit3 = circuit_from_internal(flipped, num_qbits, num_cbits)
    return circuit3

def make_times3(circ):
    qasm = circ.qasm()
    internal, num_qbits, num_cbits = internal_circuit_from_qasm(qasm)
    flipped, num_qbits, num_cbits = times3(internal, num_qbits, num_cbits)
    circuit3 = circuit_from_internal(flipped, num_qbits, num_cbits)
    return circuit3

def make_times5(circ):
    qasm = circ.qasm()
    internal, num_qbits, num_cbits = internal_circuit_from_qasm(qasm)
    flipped, num_qbits, num_cbits = times5(internal, num_qbits, num_cbits)
    circuit5 = circuit_from_internal(flipped, num_qbits, num_cbits)
    return circuit5
            
