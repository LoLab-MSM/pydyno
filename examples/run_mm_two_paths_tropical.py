from mm_two_paths_model import model
from max_plus_consumption_production import run_tropical
import numpy as np

tspan = np.linspace(0, 50, 501)

run_tropical(model, tspan, diff_par=0.5, type_sign='consumption', sp_visualize=[0, 5])

