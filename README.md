[![Build Status](https://travis-ci.org/LoLab-VU/tropical.svg?branch=master)](https://travis-ci.org/LoLab-VU/tropical)
[![Coverage Status](https://coveralls.io/repos/github/LoLab-VU/tropical/badge.svg?branch=master)](https://coveralls.io/github/LoLab-VU/tropical?branch=master)
[![Code Health](https://landscape.io/github/LoLab-VU/tropical/master/landscape.svg?style=flat)](https://landscape.io/github/LoLab-VU/tropical/master)

# TroPy

The advent of quantitative techniques to probe biomolecular-signaling processes have led to increased use of 
mathematical models to extract mechanistic insight from complex datasets. These complex mathematical models 
can yield useful insights about intracellular signal execution but the task to identify key molecular drivers 
in signal execution, within a complex network, remains a central challenge in quantitative biology. This challenge 
is compounded by the fact that cell-to-cell variability within a cell population could yield multiple signal 
execution modes and thus multiple potential drivers in signal execution. Here we present a novel approach to 
identify signaling drivers and characterize dynamic signal processes within a network. Our method, TroPy, 
combines physical chemistry, statistical clustering, and tropical algebra formalisms to identify interactions 
that drive time-dependent behavior in signaling pathways. 
## Running TroPy

TroPy depends on PySB so the easiest  way to use it is to have a PySB model and you can simply do:
```python
# Import libraries
import numpy as np
from tropical.dynamic_signatures_range import run_tropical
from tropical.examples.double_enzymatic.mm_two_paths_model import model
from pysb.simulator.scipyode import ScipyOdeSimulator

# Run the model simulation to obtain the dynmics of the molecular species
tspan = np.linspace(0, 50, 101)
sim = ScipyOdeSimulator(model, tspan=tspan).run()

tro = run_tropical(model, simulations=sim, passengers_by='imp_nodes', diff_par=0.5)

```
