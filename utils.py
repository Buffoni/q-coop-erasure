import numpy as np
import pickle
import dimod
from dwave.system import DWaveSampler
import dwave_networkx as dnx
import networkx as nx
from itertools import combinations


def get_pegasus_subgraph(sampler, n):
    subgrp = dnx.pegasus_graph(n, coordinates=True)
    nodelist = []
    for node in list(subgrp.nodes):
        nodelist.append(dnx.pegasus_coordinates(16).pegasus_to_linear(node))
    nodelist = [n for n in nodelist if n in sampler.nodelist]
    edgelist = [e for e in sampler.edgelist if e[0] in nodelist and e[1] in nodelist]
    return edgelist, nodelist

def get_pegasus_old(sampler, n):
    nodelist = sampler.nodelist[:n]
    edgelist = [e for e in sampler.edgelist if e[0] in nodelist and e[1] in nodelist]
    return edgelist, nodelist


if __name__ == '__main__':
    mytoken = '<YOUR TOKEN>' 
    solver = 'Advantage_system6.4'
    qpu_sampler = DWaveSampler(solver=solver, token=mytoken)
    #nvals = [1, 5, 10, 50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    #num_edges = []
    #for n in nvals:
    #    edgelist, nodelist = get_pegasus_old(qpu_sampler, n)
    #    num_edges.append(len(edgelist))
    #print(nvals)
    #print(num_edges)
    n = 16
    edgelist, nodelist = get_pegasus_subgraph(qpu_sampler, n)
    #edgelist, nodelist = get_pegasus_old(qpu_sampler, n)
    print(len(nodelist))
    #print(len(edgelist))
    #print(len(edgelist)/len(nodelist))
