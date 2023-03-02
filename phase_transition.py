import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import numpy as np
import pickle

rc('text',usetex=False)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams['axes.linewidth']  = 2.0
plt.rcParams['axes.labelsize']  = 25
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 1
plt.rcParams['ytick.minor.size'] = 1
plt.rcParams['legend.fontsize']  = 22
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['legend.frameon']  = False
plt.rcParams['figure.figsize'] = 7, 5


mag_graph = []
jvals = np.arange(0.01, 0.3, 0.005)

for jval in jvals:
    with open('linear_ramp' + str(int(jval * 100)) + '.pkl', 'rb') as f:
        explog = pickle.load(f)
    mean_mag_final = [explog['final_states'][0][i].mean() for i in range(len(explog['final_states'][0]))]
    mag_graph.append(np.mean(np.abs(mean_mag_final)))

# mag_graph=mag_graph-np.mean(mag_graph[0:10])

plt.plot(jvals,mag_graph,'o')
#plt.plot([0.44,0.44],[-0.1,1],'k--')
#plt.plot([0.28,0.28],[-0.1,1],'k--')
plt.ylim(-0.1,1)
plt.ylabel(r'$|m_z|$')
plt.xlabel(r'$\mathcal{J}$')
plt.tight_layout()
#plt.savefig('critical_point.pdf')
plt.show()
