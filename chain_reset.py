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
solver = 'Advantage_system6.2'

qpu_sampler = DWaveSampler(solver=solver, token=mytoken)

hmax = 0.6

num_samples = 200  # to be multiplied by num_reads
anneal_lenght = 30  # microseconds

h_schedules = []
total_schedules = []

h_schedules.append([[0, 0], [10, 0], [20, 1], [anneal_lenght, 0], [anneal_lenght + 0.5, 0]])
total_schedules.append([[0, 1], [10, 1 - hmax], [20, 1 - hmax], [anneal_lenght, 1], [anneal_lenght + 0.5, 1]])

for n in range(2,40,2):
    h = dict(enumerate([0.0 for i in range(n**2)]))

    lattice = nx.generators.path_graph(n**2)
    topology = []
    for i in lattice.edges:
        topology.append(i)

    J = {link: -0.12 for link in topology}
    h = {node: -0.5 for node in range(n**2)}

    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
    qpu_sampler = DWaveSampler(solver=solver, token=mytoken)

    current_best = n*n
    for _ in range(200):
        embedding = find_embedding(topology, qpu_sampler.edgelist)
        if len([i for i in list(embedding.values()) if len(i) > 1]) < current_best:
            best_embedding = embedding
            current_best = len([i for i in list(embedding.values()) if len(i) > 1])
        

    explog = {
        'name': './coop/quantum_chain_' + str(n),
        'num_samples': num_samples * 10,
        'anneal_lenght': anneal_lenght,
        'N': n ** 2,
        'h': -1.,
        'J': -0.12
        'solver': solver,
        'h_schedule': h_schedules,
        'schedule': total_schedules,
        'initial_states': [],
        'final_states': [],
    }

    sampler = FixedEmbeddingComposite(qpu_sampler, best_embedding)

    init_states = []
    fin_states = []
    anneal_schedule = total_schedules[0]
    h_gain_schedule = h_schedules[0]
    for i in range(num_samples):
        initial_config = 2*np.random.randint(2, size=n**2) - 1
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
