import numpy as np
import networkx as nx
from numpy.random import rand
import matplotlib.pyplot as plt
import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
import dimod
from dimod.reference import samplers
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from minorminer import find_embedding


mytoken = 'CINE-7a7dd30e6b6196bae3c9c198ee323b7e2ea3f2ed'
solver = 'Advantage_system6.1'

N = 14
with open('embedding'+str(N)+'.pkl', 'rb') as f:
    embedding = pickle.load(f)

h = dict(enumerate([0.0 for i in range(N**2)]))
lattice = nx.generators.lattice.grid_2d_graph(N, N)
topology = []
for i in lattice.edges:
    topology.append((i[0][0]*N+i[0][1],i[1][0]*N+i[1][1]))

jvals = np.arange(0.01, 0.3, 0.005)

for jval in jvals:
    J = {link: -jval for link in topology}
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
    qpu_sampler = DWaveSampler(solver=solver, token=mytoken)

    sampler = FixedEmbeddingComposite(qpu_sampler, embedding)

    num_samples = 10  # to be multiplied by num_reads
    anneal_lenght = 200  # microseconds

    explog = {
        'name': 'linear_ramp' + str(int(jval * 100)),
        'num_samples': num_samples * 100,
        'anneal_lenght': anneal_lenght,
        'N': N ** 2,
        'h': h,
        'J': J,
        'solver': solver,
        'h_schedule': [],
        'schedule': [],
        'initial_states': [],
        'final_states': [],
    }

    fin_states = []
    for i in range(num_samples):
        samples = sampler.sample(bqm, annealing_time=anneal_lenght, num_reads=100, auto_scale=False)

        for s in samples.samples():
            fin_states.append(np.array(list(s.values())))

    explog['final_states'].append(fin_states)
    with open(explog['name'] + '.pkl', 'wb') as f:
        pickle.dump(explog, f)


