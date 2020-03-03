from qiskit import *
import numpy as np
import pylab as pl
import networkx as nx
import random as ran
from qiskit.visualization import *


def randomCircuit(G,cnots=0, depth=0): #Either cnots or depth can be zero, which means "unspecified".
    V = list(G.nodes)
    num_V = len(V)
    L = nx.line_graph(G)
    MI = nx.maximal_independent_set(L)
    
    q = QuantumRegister(num_V)
    c = ClassicalRegister(num_V)
    circ = QuantumCircuit(q,c)
    circ.barrier()
    ran.seed()
    #copy of circuit for internal tracing:
    cc = []
    for j in V:
        cc.append([])
    dd = []
    for j in V:
        dd.append(0)

    #begin construction:
    if (cnots and depth):
        if (2*cnots + 1 < depth):
            print("Impossible circuit parameters: number of CNOTs is too low to reach the desired depth. Try again with different parameters.")
        elif (depth*len(MI) < cnots):
            print("Impossible circuit parameters: depth is too low to fit all the required CNOTs into the given graph. Try again with different parameters.")
        else:
            print("Constructing circuit with ", cnots, " CNOTs and ", depth," depth...")

            #preconstruction:
            for j in V:
                for d in range(depth):
                    cc[j].append(0)
            print('Preconstruction...')
            
            k = 0
            sparse = True
            contin = True

            #adding CNOT gates:
            while (sparse):
                n = 0
                cc = []
                for j in V:
                    cc.append([])
                for j in V:
                    for d in range(depth):
                        cc[j].append(0)
                #print('new attempt:', k)
                if (k > 100*cnots*depth*depth):
                    print("Sorry, unable to construct the circuit after ", k, " attempts. Try again with different parameters.")
                    contin = False
                    break
                while (n < cnots):
                    k +=1
                    if (k > 100*cnots*depth*depth):
                        break
                    edge = ran.sample(G.edges(),1)[0]
                    node1 = edge[0]
                    node2 = edge[1]
                    d = ran.randint(0,depth-1)
                    
                    if (cc[node1][d] == 0 and cc[node2][d] == 0):
                        if ran.choice([0,1]):                   
                            cc[node1][d] = ['C', node2]
                            cc[node2][d] = 'X'
                        else:
                            cc[node2][d] = ['C', node1]
                            cc[node1][d] = 'X'
                        n +=1
                
                #check if circuit is too sparse
                unsparse = False
                for j in V:
                    sparsehere = False
                    for d in range(depth-1):
                        if (cc[j][d] == 0 and cc[j][d+1] == 0):
                            sparsehere = True
                            break
                    if not(sparsehere):
                        unsparse = True
                if (unsparse):
                    sparse = False
                
            #print('cc looks like this:')
            #print(cc)
            #actual construction:
            if (contin):
                print('Successful at attempt ', k)
                print('Constructing circuit...')
                for j in V:
                    if (isinstance(cc[j][0],list)):
                        if (cc[j][0][0] == 'C'):
                            if (cc[cc[j][0][1]][0] == 'X'):
                                circ.cx(j,cc[j][0][1])
                    elif (cc[j][0] == 0):
                        theta = ran.uniform(0.0, 2*np.pi)
                        phi = ran.uniform(0.0, 2*np.pi)
                        lam = ran.uniform(0.0, 2*np.pi)
                        circ.u3(theta, phi, lam, j)
                        
                for d in range(1,depth):
                    for j in V:
                        if (isinstance(cc[j][d],list)):
                            if (cc[j][d][0] == 'C'):
                                if (cc[cc[j][d][1]][d] == 'X'):
                                    circ.cx(j,cc[j][d][1])
                        elif ((cc[j][d] == 0) and (cc[j][d-1] != 0)):
                            theta = ran.uniform(0.0, 2*np.pi)
                            phi = ran.uniform(0.0, 2*np.pi)
                            lam = ran.uniform(0.0, 2*np.pi)
                            circ.u3(theta, phi, lam, j)
                            
                          
    elif (cnots and not(depth)):
        print("Constructing circuit with ", cnots, " CNOTs and arbitrary depth...")
        n = 0
        while (n < cnots):
            gate=ran.choice(['CX','U3']) #choose randomly between CNOT and U3
            if (gate == 'U3'):
                node = ran.choice(range(num_V))
                if (not(cc[node]) or (cc[node][-1] != 'U3')):
                    theta = ran.uniform(0.0, 2*np.pi)
                    phi = ran.uniform(0.0, 2*np.pi)
                    lam = ran.uniform(0.0, 2*np.pi)
                    circ.u3(theta, phi, lam, node)
                    cc[node].append('U3')
            else:
                n += 1
                edge = ran.sample(G.edges(),1)[0]
                node1 = edge[0]
                node2 = edge[1]
                if ran.choice([0,1]):
                    circ.cx(node1,node2)
                    cc[node1].append('C')
                    cc[node2].append('X')
                else:
                    circ.cx(node2,node1)
                    cc[node2].append('C')
                    cc[node1].append('X')
    elif (not(cnots) and depth):
        print("Constructing circuit with arbitrarly many CNOTs and ", depth," depth...")
        d = 0
        while(d < depth):
            gate=ran.choice(['CX','U3']) #choose randomly between CNOT and U3
            if (gate == 'U3'):
                node = ran.choice(range(num_V))
                if (not(cc[node]) or (cc[node][-1] != 'U3')):
                    theta = ran.uniform(0.0, 2*np.pi)
                    phi = ran.uniform(0.0, 2*np.pi)
                    lam = ran.uniform(0.0, 2*np.pi)
                    circ.u3(theta, phi, lam, node)
                    cc[node].append('U3')
                    dd[node] += 1
            else:
                edge = ran.sample(G.edges(),1)[0]
                node1 = edge[0]
                node2 = edge[1]
                if ran.choice([0,1]):
                    circ.cx(node1,node2)
                    cc[node1].append('C')
                    cc[node2].append('X')
                else:
                    circ.cx(node2,node1)
                    cc[node2].append('C')
                    cc[node1].append('X')
                dd[node1] += 1
                dd[node2] += 1
                dd[node1] = max(dd[node1],dd[node2])
                dd[node2] = dd[node1]
            d = max(dd)
            print(d)
    else:
        print("This will only return an empty circuit.")
    circ.barrier()
    circ.measure(q,c)
    return circ


V = np.arange(0,5,1)
E =[(0,1,1.0),(1,2,1.0),(2,3,1.0),(3,4,1.0),(4,5,1.0),(3,5,1.0)] 

G = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E)
#print('Edges:  ', G.edges())

#print(createCircuit(np.array((np.pi,np.pi,.6,.4,.3,.8,2.1,0)),G,2).draw(output='text'))



print(randomCircuit(G, 131, 127).draw(output='text'))

print("Done")