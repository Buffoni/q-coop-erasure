import networkx as nx
import pickle
import dimod
from dwave.system.samplers import DWaveSampler
from minorminer import find_embedding


mytoken = 'CINE-7a7dd30e6b6196bae3c9c198ee323b7e2ea3f2ed'

N = 16
solver = 'Advantage_system6.1'

h = dict(enumerate([0.0 for i in range(N**2)]))

lattice = nx.generators.lattice.grid_2d_graph(N, N)
topology = []
for i in lattice.edges:
    topology.append((i[0][0]*N+i[0][1],i[1][0]*N+i[1][1]))

J = {link: -0.15 for link in topology}

bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
qpu_sampler = DWaveSampler(solver=solver, token=mytoken)

current_best = N*N
for _ in range(200):
    embedding = find_embedding(topology, qpu_sampler.edgelist)
    if len([i for i in list(embedding.values()) if len(i) > 1]) < current_best:
        best_embedding = embedding
        current_best = len([i for i in list(embedding.values()) if len(i) > 1])

print('Best embedding found contains %d chains'%(current_best))
with open('embedding'+str(N)+'.pkl', 'wb') as f:
    pickle.dump(best_embedding, f)
