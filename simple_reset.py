import numpy as np
import networkx as nx
from numpy.random import rand
import matplotlib.pyplot as plt
import pickle
import time
import matplotlib.pyplot as plt
import dimod
from dimod.reference import samplers
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite, FixedEmbeddingComposite
from minorminer import find_embedding


mytoken = 'CINE-7a7dd30e6b6196bae3c9c198ee323b7e2ea3f2ed'
solver = 'Advantage_system6.1'

qpu_sampler = DWaveSampler(solver=solver, token=mytoken)
J = {link: 0.0 for link in qpu_sampler.edgelist}
h = {node: 1. for node in qpu_sampler.nodelist}
bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

hmax = 0.6

num_samples = 200  # to be multiplied by num_reads
anneal_lenght = 30  # microseconds

h_schedules = []
total_schedules = []

for i in range(0, 30, 1):
    if 0 < i <= 10:
        h_schedules.append([[0, 0], [i, 0], [i + 0.5, 0]])
        total_schedules.append([[0, 1], [i, (1 - i * hmax / 10)], [i + 0.5, 1]])
    elif 10 < i <= 20:
        h_schedules.append([[0, 0], [10, 0], [i, (i - 10) / 10], [i + 0.5, 0]])
        total_schedules.append([[0, 1], [10, 1 - hmax], [i, 1 - hmax], [i + 0.5, 1]])
    elif 20 < i < 30:
        h_schedules.append([[0, 0], [10, 0], [20, 1], [i, 1 - (i - 20) / 10], [i + 0.5, 0]])
        total_schedules.append([[0, 1], [10, 1 - hmax], [20, 1 - hmax], [i, (1 - hmax + (i - 20) * hmax / 10)], [i + 0.5, 1]])

h_schedules.append([[0, 0], [10, 0], [20, 1], [anneal_lenght, 0], [anneal_lenght + 0.5, 0]])
total_schedules.append([[0, 1], [10, 1 - hmax], [20, 1 - hmax], [anneal_lenght, 1], [anneal_lenght + 0.5, 1]])

explog = {
    'name': './individual/individual_quantum',
    'num_samples': num_samples * 10,
    'anneal_lenght': anneal_lenght,
    'N': qpu_sampler.properties["qubits"],
    'h': h,
    'J': J,
    'solver': solver,
    'h_schedule': [],
    'schedule': [],
    'initial_states': [],
    'final_states': [],
}

for k in range(len(total_schedules)):
    init_states = []
    fin_states = []
    anneal_schedule = total_schedules[k]
    h_gain_schedule = h_schedules[k]
    for i in range(num_samples):
        initial_config = 2*np.random.randint(2, size=len(qpu_sampler.properties["qubits"])) - 1
        initial_state = {qpu_sampler.properties["qubits"][i]: initial_config[i]
                         for i in range(len(qpu_sampler.properties["qubits"]))}
        samples = qpu_sampler.sample(bqm, initial_state=initial_state,
                                    anneal_schedule=anneal_schedule,
                                    h_gain_schedule=h_gain_schedule,
                                    num_reads=10, auto_scale=False)

        for s in samples.samples():
            fin_states.append(np.array(list(s.values())))
            init_states.append(initial_config)

    explog['final_states'].append(fin_states)
    explog['initial_states'].append(init_states)

with open(explog['name']+'.pkl', 'wb') as f:
    pickle.dump(explog, f)

