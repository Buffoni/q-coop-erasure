import numpy as np
import pickle
import dimod
from dwave.system import DWaveSampler


def get_pegasus_subgraph(sampler, n):
    nodelist = sampler.nodelist[:n]
    edgelist = [e for e in sampler.edgelist if e[0] in nodelist and e[1] in nodelist]
    return edgelist, nodelist


if __name__ == '__main__':
    mytoken = 'CINE-7a7dd30e6b6196bae3c9c198ee323b7e2ea3f2ed'
    solver = 'Advantage_system6.4'
    qpu_sampler = DWaveSampler(solver=solver, token=mytoken)
    n = 10
    edgelist, nodelist = get_pegasus_subgraph(qpu_sampler, n)
    print(nodelist)
    print(edgelist)