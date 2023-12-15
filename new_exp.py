import numpy as np
import pickle
import dimod
from dwave.system import DWaveSampler

mytoken = 'DEV-2942a9351f40088a2e32f4f1732b5dd8dcffea46' # michele
# mytoken = 'DEV-cd1ac920efa5f032dd82604d3bd6544669d486dc' # lorenzo
solver = 'Advantage_system6.3'

qpu_sampler = DWaveSampler(solver=solver, token=mytoken)
nval = len(qpu_sampler.edgelist)

#jvals = np.arange(0, 0.1, 0.005)
jvals = [0]

num_samples = 100  # to be multiplied by num_reads
#anneal_lenght = 100  # microseconds
schedule = [[0, 1], [1,0.9],[2,1]]

for jval in jvals:
    explog = {
        'name': 'new_reset_noprotocol',
        'num_samples': num_samples * 10,
        #'anneal_lenght': anneal_lenght,
        'anneal_schedule': schedule,
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
        initial_state = {node: 1. for node in qpu_sampler.nodelist}
        #samples = qpu_sampler.sample(bqm, annealing_time=anneal_lenght, num_reads=10, auto_scale=False)
        samples = qpu_sampler.sample(bqm, initial_state=initial_state, anneal_schedule=schedule, num_reads=10, auto_scale=False)

        for s in samples.samples():
            fin_states.append(np.array(list(s.values())))

    explog['final_states'].append(fin_states)

    with open(explog['name']+ '.pkl', 'wb') as f:  # str(int(jval*1000))+
        pickle.dump(explog, f)