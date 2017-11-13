import numpy as np

from miscellaneous_tools.miscellaneous_analysis import change_parameter_in_time
from pysb.examples.tyson_oscillator import model

tspan = np.linspace(0, 200, 200)

change_parameter_in_time(model, tspan, 51, 5, ['kp4'], 2)
