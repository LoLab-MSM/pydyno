from tropical.dynamic_signatures_range import run_tropical_multi
from mm_two_paths_model import model
import pickle

signatures = run_tropical_multi(model=model, simulations='sim_results.h5', cpu_cores=4)
with open('signatures2.pickle', 'wb') as handle:
    pickle.dump(signatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
