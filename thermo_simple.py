import numpy as np
import pickle
import dimod
from dwave.system.samplers import DWaveSampler

mytoken = '<YOUR TOKEN>' 
solver = 'Advantage_system6.1'

qpu_sampler = DWaveSampler(solver=solver, token=mytoken)

hvals = [0, 0.1, 0.5, 1, 1.5, 2]

num_samples = 200  # to be multiplied by num_reads
anneal_lenght = 100  # microseconds

explog = {
    'name': './simple_thermometer',
    'num_samples': num_samples * 10,
    'anneal_lenght': anneal_lenght,
    'N': qpu_sampler.properties["qubits"],
    'h': hvals,
    'solver': solver,
    'final_states': [],
}

for hval in hvals:
    J = {link: 0.0 for link in qpu_sampler.edgelist}
    h = {node: hval for node in qpu_sampler.nodelist}
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

    fin_states = []

    for i in range(num_samples):
        samples = qpu_sampler.sample(bqm, anneal_lenght=anneal_lenght,
                                    num_reads=10, auto_scale=False)

        for s in samples.samples():
            fin_states.append(np.array(list(s.values())))

    explog['final_states'].append(fin_states)

with open(explog['name']+'.pkl', 'wb') as f:
    pickle.dump(explog, f)
