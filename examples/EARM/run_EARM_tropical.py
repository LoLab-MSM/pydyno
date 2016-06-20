import matplotlib
from earm.lopez_embedded import model
from tropicalize import run_tropical
import numpy as np
import csv
from pysb import *
# tipe Bax cluster: 5400
# type Bak cluster: 4052

f = open('/home/oscar/Documents/tropical_project/parameters_5000/pars_embedded_4052.txt')
data = csv.reader(f)
parames = [float(i[1]) for i in data]
t = np.linspace(0, 20000,  100)

run_tropical(model, t, parames, sp_visualize=[19])

# import matplotlib.pyplot as plt
# from pysb.integrate import odesolve
#
# y = odesolve(model, t, parames)
#
# p = model.parameters
# index_kf_bak = p.index(p['pore_transport_complex_BakA_4_SmacM_kf'])
# index_kf_bax = p.index(p['pore_transport_complex_BaxA_4_SmacM_kf'])
# index_kr_bak = p.index(p['pore_transport_complex_BakA_4_SmacM_kr'])
# index_kr_bax = p.index(p['pore_transport_complex_BaxA_4_SmacM_kr'])
#
# plt.figure(figsize=(12, 8))
# plt.plot(t, y['__s20']*y['__s60']*parames[index_kf_bak], linewidth=2, label='s20*s60*pore_transport_complex_BakA_4_SmacM_kf')
# plt.plot(t, y['__s20']*y['__s64']*parames[index_kf_bax], linewidth=2, label='s20*s64*pore_transport_complex_BaxA_4_SmacM_kf')
# plt.plot(t, y['__s63']*parames[index_kr_bak], linewidth=2, label='s63*pore_transport_complex_BakA_4_SmacM_kr')
# plt.plot(t, y['__s70']*parames[index_kr_bax], linewidth=2, label='s70*pore_transport_complex_BaxA_4_SmacM_kr')
# plt.xlabel('Time(sec)', fontsize=18)
# plt.ylabel('molecules', fontsize=18)
# plt.title('SmacM monomials', fontsize=20)
# plt.rc('legend',**{'fontsize':12})
# plt.legend(loc=0)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.savefig('/home/oscar/smac_bak.png', dpi=100, bbox_inches='tight')
# plt.show()



