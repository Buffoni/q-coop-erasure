import numpy as np
import pickle
import dimod
from dwave.system import DWaveSampler
from utils import get_pegasus_subgraph
import csv

mytoken = 'CINE-7a7dd30e6b6196bae3c9c198ee323b7e2ea3f2ed'
solver = 'Advantage_system6.4'

qpu_sampler = DWaveSampler(solver=solver, token=mytoken)
nval = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

hmax = 0.65
anneal_lenght = [10,20,30]  # microseconds


for n in nval:
    for lenght in anneal_lenght:
        h_schedules = []
        total_schedules = []

        h_schedules.append([[0, 0], [lenght / 3, 0], [2 * lenght / 3, 1], [lenght, 1], [lenght + 0.01, 0]])
        total_schedules.append([[0, 1], [lenght / 3, 1 - hmax], [2 * lenght / 3, 1 - hmax], [lenght, 1], [lenght + 0.01, 1]])

        edgelist, nodelist = get_pegasus_subgraph(qpu_sampler, nval)

        explog = {
            'name': 'battery_charge_T'+str(lenght)+'_N'+str(n),
            'num_samples': 1000,
            'anneal_lenght': anneal_lenght,
            'N': n,
            'h': -2.,
            'J': -0.2,
            'solver': solver,
            'h_schedule': total_schedules,
            'schedule': h_schedules,
            'initial_states': [],
            'final_states': [],
        }

        J = {link: -0.2 for link in edgelist} #-0.2
        h = {node: -2. for node in nodelist}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        init_states = []
        fin_states = []
        anneal_schedule = total_schedules[0]
        h_gain_schedule = h_schedules[0]
        initial_config = - np.ones(n)
        initial_state = {nodelist[i]: initial_config[i] for i in range(len(nodelist))}
        samples = qpu_sampler.sample(bqm, initial_state=initial_state,
                                    anneal_schedule=anneal_schedule,
                                    h_gain_schedule=h_gain_schedule,
                                    answer_mode='raw',
                                    num_reads=1000, auto_scale=False)

        for s in samples.samples():
            fin_states.append(np.array(list(s.values())))
            init_states.append(initial_config)

        explog['final_states'] = fin_states
        explog['initial_states'] = init_states

        with open(explog['name']+'.csv', 'wb') as f:
            wr = csv.writer(f)
            wr.writerow(explog['final_states'])

        with open(explog['name']+'.pkl', 'wb') as f:
            pickle.dump(explog, f)