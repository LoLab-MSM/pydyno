from earm.lopez_embedded import model 
# from pysb.examples.earm_1_0 import model
# from max_monomials_signature import run_tropical
from tropicalize import run_tropical
import numpy as np
import matplotlib.pyplot as plt
import csv

f = open('/home/carlos/Downloads/pars_embedded_911.txt') 
data = csv.reader(f)
parames = []
for i in data:parames.append(float(i[1]))

t = np.linspace(0,20000,100)

print run_tropical(model,t, parames)

#486 [1, 17, 18, 21, 22, 23, 24, 26, 27, 31, 32, 34, 35, 36, 37, 41, 42, 46, 48, 49, 50, 51, 52, 53, 59, 60, 61, 62, 63, 65, 66, 68, 69, 70, 71, 75]
#911 [0, 1, 17, 21, 22, 25, 26, 27, 31, 34, 36, 41, 42, 43, 46, 48, 49, 50, 51, 54, 55, 56, 58, 59, 61, 62, 63, 65, 66, 68, 69, 70, 71, 75]
