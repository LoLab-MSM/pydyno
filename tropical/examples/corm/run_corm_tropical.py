from tropical.discretize import Discretize
from tropical.examples.corm.corm import model

disc = Discretize(model, simulations='corm_unique_trajectories.h5', diff_par=1)
disc.get_signatures()
