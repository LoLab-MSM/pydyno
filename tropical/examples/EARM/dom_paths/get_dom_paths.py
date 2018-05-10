from tropical.dominant_path_analysis import run_dompath_multi, run_dompath_single
import pickle

sims_path = 'earm_scipyode_sims.h5'

signatures = run_dompath_multi(sims_path, target='s39', depth=6, cpu_cores=1, verbose=True)
# signatures = run_dompath_single(sims_path, target='s39', depth=12)

with open('path_signatures.pickle', 'wb') as handle:
    pickle.dump(signatures, handle, protocol=pickle.HIGHEST_PROTOCOL)