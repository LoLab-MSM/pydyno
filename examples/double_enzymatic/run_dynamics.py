from mm_two_paths_model import model
import numpy as np
import matplotlib.pyplot as plt

tspan = np.linspace(0,50,100)

tspan = np.linspace(0, 20, 100)
y = odesolve(model, tspan)

color = ['b', 'r', 'b', 'g']

for i, j in enumerate(model.observables):
    plt.figure()
    plt.plot(tspan, y[j.name], label=j.name, linewidth=4, color=color[i])
    plt.xlabel('time', fontsize=20)
    plt.ylabel('Population', fontsize=20)
    plt.legend(loc=0)
    plt.savefig('/home/oscar/Desktop/mm%d' % i + '.eps')
plt.show()
