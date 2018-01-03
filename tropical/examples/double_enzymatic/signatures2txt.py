from tropical.tools.miscellaneous_analysis import trajectories_signature_2_txt
from tropical.examples.double_enzymatic.mm_two_paths_model import model
import numpy as np

tspan = np.linspace(0, 10, 51)

trajectories_signature_2_txt(model=model, tspan=tspan, sp_to_analyze=[0])