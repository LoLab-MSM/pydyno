[![Build Status](https://travis-ci.org/LoLab-VU/tropical.svg?branch=master)](https://travis-ci.org/LoLab-VU/tropical)
[![Coverage Status](https://coveralls.io/repos/github/LoLab-VU/tropical/badge.svg?branch=master)](https://coveralls.io/github/LoLab-VU/tropical?branch=master)
[![Code Health](https://landscape.io/github/LoLab-VU/tropical/master/landscape.svg?style=flat)](https://landscape.io/github/LoLab-VU/tropical/master)

# TroPy

We present TroPy, a novel approach to study cellular signaling networks from a dynamic perspective. This method combines the Quasi-Steady State approach with max-plus algebra to find the driver species and the specific reactions that contribute the most to species concentration changes in time. Hence, it is possible to study how those driver species and reactions change for different parameter sets that fit the data equally well. Finally, it is possible to identify clusters of parameter that generate similar modes of signal execution in signal transduction pathways.

## Running TroPy

TroPy depends on PySB so the easiest  way to use it is to have a PySB model and you can simply do:
```python
from mymodel import model
from tropicalize import Tropical
import numpy as np

tr = Tropical(model)
tspan = np.linspace(0,200, 200)

#This commands runs the QSSA and the tropical algebra tools and identify the drivers and passenger species
tr.tropicalize(tspan)

#To get the driver species run:
tr.get_drivers()

#To visualize the dominant interactions at each time point por each species run:
tr.visualization(tspan, parameters)
```
