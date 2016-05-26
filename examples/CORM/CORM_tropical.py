from tropicalize import run_tropical
from corm import model
import numpy


tspan = numpy.linspace(0, 10, num=100)
run_tropical(model, tspan,   sp_visualize=[3, 4])

# tr = Tropical(model)
# tr.tropicalize(tspan)
# print tr.get_passenger()
