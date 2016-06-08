from pysb.examples.tyson_oscillator import model
from tropicalize import Tropical
from numpy import linspace
from nose.tools import *


t = linspace(0, 200, 200)
tro = Tropical(model)
tro.tropicalize(t)


def test_slaves():
    assert_equal(tro.passengers, [0, 1, 4])

test_slaves()
