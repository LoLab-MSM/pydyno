from tropical.discretize import Discretize
from tropical.examples.double_enzymatic.mm_two_paths_model import model

disc = Discretize(model, 'sim_results.h5')
signatures = disc.get_signatures(cpu_cores=4)

