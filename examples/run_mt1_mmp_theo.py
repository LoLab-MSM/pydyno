import mt1_mmp_model
from analyze_mt1_mmp import run_solve_theo
import numpy

model_mt1 = mt1_mmp_model.return_model('original')

t2 = numpy.linspace(0, 5, 500)

run_solve_theo(model_mt1, t2)