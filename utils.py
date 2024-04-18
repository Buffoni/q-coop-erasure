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
    # return subgrp.edges, subgrp.nodes as lists
    return edgelist, nodelist


if __name__ == '__main__':
    mytoken = 'CINE-7a7dd30e6b6196bae3c9c198ee323b7e2ea3f2ed'
    solver = 'Advantage_system6.4'
    qpu_sampler = DWaveSampler(solver=solver, token=mytoken)
    n = 3
    edgelist, nodelist = get_pegasus_subgraph(qpu_sampler, n)
    #edgelist, nodelist = highest_connectivity_subgraph(qpu_sampler.nodelist, qpu_sampler.edgelist, n)
    print(len(nodelist))
    #print(edgelist)
    print(len(edgelist)/len(nodelist))