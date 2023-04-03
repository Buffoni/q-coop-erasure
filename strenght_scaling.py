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

hvals = np.linspace(0,2,20)
hmax = 0.4

num_samples = 200  # to be multiplied by num_reads
anneal_lenght = 30  # microseconds

h_schedules = []
total_schedules = []

h_schedules.append([[0, 0], [10, 0], [20, 1], [anneal_lenght, 0], [anneal_lenght + 0.5, 0]])
total_schedules.append([[0, 1], [10, 1 - hmax], [20, 1 - hmax], [anneal_lenght, 1], [anneal_lenght + 0.5, 1]])

for n in range(3,19):
    lattice = nx.generators.lattice.grid_2d_graph(n, n)
    topology = []
    for i in lattice.edges:
        topology.append((i[0][0] * n + i[0][1], i[1][0] * n + i[1][1]))

    explog = {
        'name': './individual/classical_coop_' + str(n) + '_scaleh',
        'num_samples': num_samples * 10,
        'anneal_lenght': anneal_lenght,
        'N': n ** 2,
        'h': hvals,
        'solver': solver,
        'h_schedule': [],
        'schedule': [],
        'initial_states': [],
        'final_states': [],
    }

    for k in hvals:
        J = {link: -0.9 for link in topology}
        h = {node: -k for node in range(n**2)}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        with open('embedding' + str(n) + '.pkl', 'rb') as f:
            embedding = pickle.load(f)

        sampler = FixedEmbeddingComposite(qpu_sampler, embedding)

        init_states = []
        fin_states = []
        anneal_schedule = total_schedules[0]
        h_gain_schedule = h_schedules[0]
        for i in range(num_samples):
            initial_config = 2*np.random.randint(2, size=n**2) - 1
            #initial_state = {qpu_sampler.properties["qubits"][i]: initial_config[i]
            #                 for i in range(len(qpu_sampler.properties["qubits"]))}
            initial_state = dict(enumerate(initial_config.tolist()))
            samples = sampler.sample(bqm, initial_state=initial_state,
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