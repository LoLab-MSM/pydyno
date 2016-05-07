from pysb.examples.tyson_oscillator import model
from tropicalize import run_tropical
from numpy import linspace
from nose.tools import *
import traceback
import os
import importlib

t = linspace(0, 200, 200)
tro = run_tropical(model, t)


def test_slaves():
    assert_equal(tro.passengers, [0, 1, 4])
