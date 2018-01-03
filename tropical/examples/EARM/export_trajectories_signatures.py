from tropical.tools.miscellaneous_analysis import trajectories_signature_2_txt
import numpy as np
from earm2_flat import model

tspan = np.linspace(0, 20000, 100)
trajectories_signature_2_txt(model, tspan=tspan, sp_to_analyze=[37], parameters=None)
