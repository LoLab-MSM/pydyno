from mm_two_paths_model import model
import numpy as np
import matplotlib.pyplot as plt
from pysb.simulator import ScipyOdeSimulator
tspan = np.linspace(0,50,100)

tspan = np.linspace(0, 20, 100)
y = ScipyOdeSimulator(model, tspan).run().dataframe

color = ['b', 'r', 'c', 'g', 'k', 'y']

plt.figure()
for i, j in enumerate(model.observables):
    plt.plot(tspan, y[j.name], label=j.name, linewidth=4, color=color[i], alpha=0.5)
    plt.xlabel('time', fontsize=20)
    plt.ylabel('Population', fontsize=20)
    plt.legend(loc=0)
    plt.savefig('/Users/dionisio/Desktop/mm%d' % i + '.eps')
plt.show()
