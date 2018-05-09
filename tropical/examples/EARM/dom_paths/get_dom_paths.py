from tropical.dominant_path_analysis import run_dompath_multi
import pickle

sims_path = 'earm_scipyode_sims.h5'

signatures = run_dompath_multi(sims_path, target='s39', depth=6, cpu_cores=25)

with open('path_signatures.pickle', 'wb') as handle:
    pickle.dump(signatures, handle, protocol=pickle.HIGHEST_PROTOCOL)