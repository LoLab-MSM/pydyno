from tropical.helper_functions import trajectories_signature_2_txt
import numpy as np
import os
from earm.lopez_embedded import model

directory = os.path.dirname(__file__)
pars_path = os.path.join(directory, 'all_IC_simulations/IC_10000_pars_consumption_kpar0.npy')

all_parameters = np.load(pars_path)
tspan = np.linspace(0, 20000, 100)
trajectories_signature_2_txt(model, tspan=tspan, sp_to_analyze=[37], parameters=all_parameters[1643],
                             file_path='/Users/dionisio/Desktop/')
