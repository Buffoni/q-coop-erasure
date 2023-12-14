import numpy as np
import networkx as nx
from numpy.random import rand
import matplotlib.pyplot as plt
import pickle
import time
import matplotlib.pyplot as plt
import dimod
from dimod.reference import samplers
from dwave.system import DWaveSampler
from dwave.system import EmbeddingComposite, FixedEmbeddingComposite
from minorminer import find_embedding

mytoken = 'DEV-2942a9351f40088a2e32f4f1732b5dd8dcffea46' # michele
# mytoken = 'DEV-cd1ac920efa5f032dd82604d3bd6544669d486dc' # lorenzo
solver = 'Advantage_system6.3'

qpu_sampler = DWaveSampler(solver=solver, token=mytoken)
nval = len(qpu_sampler.edgelist)

jvals = np.arange(0, 0.1, 0.005)

num_samples = 10  # to be multiplied by num_reads
anneal_lenght = 100  # microseconds

for jval in jvals:
    explog = {
        'name': 'linear_ramp_pegasus',
        'num_samples': num_samples * 10,
        'anneal_lenght': anneal_lenght,
        'N': nval,
        'h':0,
        'J': jval,
        'solver': solver,
        'final_states': [],
    }

    J = {link: -jval for link in qpu_sampler.edgelist}
    h = {node: 0 for node in qpu_sampler.nodelist}
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

    fin_states = []
    for i in range(num_samples):
        samples = qpu_sampler.sample(bqm, annealing_time=anneal_lenght,
                                    num_reads=10, auto_scale=False)

        for s in samples.samples():
            fin_states.append(np.array(list(s.values())))

    explog['final_states'].append(fin_states)

    with open(explog['name']+ str(int(jval * 100)) + '.pkl', 'wb') as f:
        pickle.dump(explog, f)