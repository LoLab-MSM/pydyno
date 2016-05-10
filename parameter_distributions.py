import matplotlib.pyplot as plt
import matplotlib
import helper_functions as hf
matplotlib.style.use('ggplot')


def parameter_distribution(parameters_path, par_name):
    all_parameters = hf.read_all_pars(parameters_path)
    plt.figure()
    all_parameters[par_name].plot.hist()
    plt.show()
