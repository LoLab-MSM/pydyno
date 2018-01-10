from tropical.dynamic_signatures_range import run_tropical_multi
from tropical.examples.corm.corm import model
import pickle

a = run_tropical_multi(model, simulations='corm_unique_trajectories.h5', cpu_cores=11)

with open('earm_signatures.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)