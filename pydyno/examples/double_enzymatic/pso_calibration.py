import numpy as np
from pydyno.examples.double_enzymatic.mm_two_paths_model import model
from simplepso.pso import PSO
from pysb.simulator import ScipyOdeSimulator
import os
import matplotlib.pyplot as plt

directory = os.path.dirname(__file__)
avg_data_path = os.path.join(directory, 'product_data.npy')
sd_data_path = os.path.join(directory, 'exp_sd.npy')

exp_avg = np.load(avg_data_path)
exp_sd = np.load(sd_data_path)

tspan = np.linspace(0, 10, 51)

solver = ScipyOdeSimulator(model, tspan=tspan)

idx_pars_calibrate = [0, 1, 2, 3, 4, 5]
rates_mask = [i in idx_pars_calibrate for i, par in enumerate(model.parameters)]

param_values = np.array([p.value for p in model.parameters])
nominal_values = np.array([p.value for p in model.parameters])
xnominal = np.log10(nominal_values[rates_mask])
bounds_radius = 2


def display(position):
    Y = np.copy(position)
    param_values[rates_mask] = 10 ** Y
    sim = solver.run(param_values=param_values).all
    plt.plot(tspan, sim['Product'], label='Produc sim')
    plt.errorbar(tspan, exp_avg, yerr=exp_sd)
    plt.show()


def likelihood(position):
    Y = np.copy(position)
    param_values[rates_mask] = 10 ** Y
    sim = solver.run(param_values=param_values).all
    e1 = np.sum((exp_avg - sim['Product']) ** 2 / (2 * exp_sd)) / len(exp_avg)
    return e1,


def run_example():
    pso = PSO(save_sampled=False, verbose=True, num_proc=4)
    pso.set_cost_function(likelihood)
    pso.set_start_position(xnominal)
    pso.set_bounds(2)
    pso.set_speed(-0.25, 0.25)
    pso.run(25, 100)
    display(pso.best)

if __name__ == '__main__':
    run_example()
