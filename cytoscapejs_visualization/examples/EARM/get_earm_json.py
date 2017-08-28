from earm.lopez_embedded import model
import numpy
import os
from pysb.tools.cytoscapejs_visualization.model_visualization_cytoscapejs import FluxVisualization
import csv


def read_pars(par_path):
    """
    Reads parameter file
    :param par_path: path to parameter file
    :return: Return a list of parameter values from csv file
    """
    f = open(par_path)
    data = csv.reader(f)
    param = [float(d[1]) for d in data]
    return param

directory = os.path.dirname(__file__)
parameter_path = os.path.join(directory, "pars_embedded_26.txt")
parameters = read_pars(parameter_path)
t = numpy.linspace(0, 20000, 100)
a = FluxVisualization(model)
a.setup_info(tspan=t, param_values=parameters)
g_layout = a.dot_layout()
a.graph_to_json(layout=g_layout)
