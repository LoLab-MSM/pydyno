from max_monomials_signature import run_tropical
from earm.lopez_embedded import model
from pysb.integrate import odesolve
import numpy as np
import matplotlib.pyplot as plt
import csv

tspan = np.linspace(0,20000,100)

f = open('/home/carlos/Downloads/pars_embedded_911.txt') 
data = csv.reader(f)
parames = []
for i in data:parames.append(float(i[1]))

epsilons = np.linspace(0,4,80)
num_d = []
for i in epsilons:
    num_d.append(len(run_tropical(model,tspan,i,parames)[2]))
    
plt.plot(epsilons,num_d)
plt.xlabel('epsilon')
plt.ylabel('passenger species')
plt.show()
                 

