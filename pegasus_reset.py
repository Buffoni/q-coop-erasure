import numpy as np
import pickle
import dimod
from dwave.system import DWaveSampler
from utils import get_pegasus_subgraph


#mytoken = 'DEV-2942a9351f40088a2e32f4f1732b5dd8dcffea46' # michele
mytoken = 'CINE-7a7dd30e6b6196bae3c9c198ee323b7e2ea3f2ed'
solver = 'Advantage_system6.4'

qpu_sampler = DWaveSampler(solver=solver, token=mytoken)
nval = len(qpu_sampler.nodelist) # 1000
edgelist, nodelist = get_pegasus_subgraph(qpu_sampler, nval)

#hvals = np.linspace(0,2,20)
hvals = [2.] #2.0
hmax = 0.65

num_samples = 1  # to be multiplied by num_reads
anneal_lenght = 30.1  # microseconds

h_schedules = []
total_schedules = []

#h_schedules.append([[0, 0], [10, 0], [20, 1], [anneal_lenght, 0], [anneal_lenght + 0.5, 0]])
#total_schedules.append([[0, 1], [10, 1 - hmax], [20, 1 - hmax], [anneal_lenght, 1], [anneal_lenght + 0.5, 1]])
h_schedules.append([[0, 0], [10, 0], [20, 1], [30, 1], [30.01, 0]])
total_schedules.append([[0, 1], [10, 1 - hmax], [20, 1 - hmax], [30, 1], [30.01, 1]])
#total_schedules.append([[0, 1], [30, 1]])
#h_schedules.append([[0, 0], [30, 0]])



for n in [nval]:#range(3,19):

    explog = {
        'name': 'battery_charge_T30',
        'num_samples': num_samples * 1000,
        'anneal_lenght': anneal_lenght,
        'N': n,
        'h': hvals,
        'solver': solver,
        'h_schedule': total_schedules,
        'schedule': h_schedules,
        'initial_states': [],
        'final_states': [],
    }

    for k in hvals:
        J = {link: -0.2 for link in edgelist} #-0.2
        h = {node: -k for node in nodelist}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        init_states = []
        fin_states = []
        anneal_schedule = total_schedules[0]
        h_gain_schedule = h_schedules[0]
        for i in range(num_samples):
            #initial_config = 2*np.random.randint(2, size=n) - 1
            initial_config = - np.ones(n)
            #initial_state = {qpu_sampler.properties["qubits"][i]: initial_config[i]
            #                for i in range(len(qpu_sampler.properties["qubits"]))}
            initial_state = {nodelist[i]: initial_config[i]
                             for i in range(len(nodelist))}
            samples = qpu_sampler.sample(bqm, initial_state=initial_state,
                                        anneal_schedule=anneal_schedule,
                                        h_gain_schedule=h_gain_schedule,
                                        answer_mode='raw',
                                        num_reads=1000, auto_scale=False)

            for s in samples.samples():
                fin_states.append(np.array(list(s.values())))
                init_states.append(initial_config)

        explog['final_states'].append(fin_states)
        explog['initial_states'].append(init_states)

    with open(explog['name']+'.pkl', 'wb') as f:
        pickle.dump(explog, f)