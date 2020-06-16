from pydyno.tools.miscellaneous_analysis import simulate_changing_parameter_in_time
import numpy as np
from pysb.examples.tyson_oscillator import model as tyson_model
from pysb.simulator import ScipyOdeSimulator
import pytest


@pytest.fixture(scope="function")
def simulate_tyson():
    time = np.linspace(0, 100, 100)
    sim = ScipyOdeSimulator(tyson_model, tspan=time).run()
    return sim


@pytest.fixture(scope="function")
def simulate_tyson_changing_parameter():
    time = np.linspace(0, 100, 100)
    previous_parameters = np.array([p.value for p in tyson_model.parameters])
    sim = simulate_changing_parameter_in_time(model=tyson_model, tspan=time, time_change=20,
                                              previous_parameters=previous_parameters,
                                              new_parameters=previous_parameters/2)
    return sim


def test_simulate_parameter_change_in_time(simulate_tyson, simulate_tyson_changing_parameter):
    assert np.array_equal(simulate_tyson.species[:20, :], simulate_tyson_changing_parameter.species[:20, :])
    assert not np.array_equal(simulate_tyson.species[20:, :], simulate_tyson_changing_parameter.species[20:, :])