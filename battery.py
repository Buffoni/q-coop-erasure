import numpy as np
import pickle
import dimod
from dwave.system import DWaveSampler
from utils import get_pegasus_subgraph, get_pegasus_old

mytoken = 'CINE-7a7dd30e6b6196bae3c9c198ee323b7e2ea3f2ed'
solver = 'Advantage_system6.4'
qpu_sampler = DWaveSampler(solver=solver, token=mytoken)

is_coop = False

nval = [1,2,4,6,8,10,12,14,16]
hmax = 0.65
lenght = 60  # microseconds
jval = 0.2
hval = 1.

for n in nval:
    h_schedules = [[0, 0], [lenght / 3, 0], [2 * lenght / 3, 1], [lenght, 0]]
    total_schedules = [[0, 1], [lenght / 3, 1 - hmax], [2 * lenght / 3, 1 - hmax], [lenght, 1]]

    if n==1:
        edgelist, nodelist = get_pegasus_old(qpu_sampler, n)
    else:
        edgelist, nodelist = get_pegasus_subgraph(qpu_sampler, n)

    if is_coop==True:
        hscaled = hval
        jscaled = jval
        name = 'battery_T'+str(lenght)+'_N'+str((len(nodelist))) + '_coop'
    else:
        hscaled = np.sqrt(hval**2 + len(edgelist)*(0.2)**2/len(nodelist))
        jscaled = 0.
        name = 'battery_T'+str(lenght)+'_N'+str((len(nodelist)))+'_no_coop'

    explog = {
        'name': name,
        'num_samples': 1000,
        'anneal_lenght': lenght,
        'N': len(nodelist),
        'h': -hscaled,
        'J': -jscaled,
        'solver': solver,
        'h_schedule': total_schedules,
        'schedule': h_schedules,
        'initial_states': [],
        'final_states': [],
        'num_edges': len(edgelist),
    }

    J = {link: -jscaled for link in edgelist}
    h = {node: -hscaled for node in nodelist}
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

    initial_config = - np.ones(len(nodelist))
    initial_state = {nodelist[i]: initial_config[i] for i in range(len(nodelist))}
    samples = qpu_sampler.sample(bqm, initial_state=initial_state,
                                anneal_schedule=total_schedules,
                                h_gain_schedule=h_schedules,
                                answer_mode='raw',
                                num_reads=1000, auto_scale=False)

    for s in samples.samples():
        explog['final_states'].append(np.array(list(s.values())))
        explog['initial_states'] .append(initial_config)

    with open(explog['name']+'.pkl', 'wb') as f:
        pickle.dump(explog, f)
