import numpy as np
import pickle
import dimod
from dwave.system import DWaveSampler
import dwave_networkx as dnx
import networkx as nx
from itertools import combinations


def get_pegasus_subgraph(sampler, n):
    G = nx.Graph()
    G.add_nodes_from(sampler.nodelist)
    G.add_edges_from(sampler.edgelist)
    subgrp = dnx.pegasus_graph(n, create_using=G)
    # return subgrp.edges, subgrp.nodes as lists
    return list(subgrp.edges), list(subgrp.nodes)


if __name__ == '__main__':
    mytoken = 'CINE-7a7dd30e6b6196bae3c9c198ee323b7e2ea3f2ed'
    solver = 'Advantage_system6.4'
    qpu_sampler = DWaveSampler(solver=solver, token=mytoken)
    n = 15
    edgelist, nodelist = get_pegasus_subgraph(qpu_sampler, n)
    #edgelist, nodelist = highest_connectivity_subgraph(qpu_sampler.nodelist, qpu_sampler.edgelist, n)
    print(len(nodelist))
    #print(edgelist)
    print(len(edgelist)/len(nodelist))