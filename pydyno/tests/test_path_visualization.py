import sys
from pydyno.examples.double_enzymatic.mm_two_paths_model import model as enzyme_model
from pysb.simulator import ScipyOdeSimulator
import pytest
import numpy as np
import pydyno.discretization.pysb_discretize as dp
from pydyno.visualize_discretization import visualization_path


@pytest.fixture(scope="class")
def pysb_dom_path():
    time = np.linspace(0, 100, 100)
    sim = ScipyOdeSimulator(enzyme_model, tspan=time).run()
    dom = dp.PysbDomPath(model=enzyme_model, simulations=sim)
    return dom


class TestVisualizationPysbSingle:
    @pytest.mark.skipif(sys.platform == 'win32', reason="Graphviz has troubles on windows")
    def test_path_visualization(self, pysb_dom_path, data_files_dir):
        signatures, paths = pysb_dom_path.get_path_signatures(type_analysis='consumption',
                                                              dom_om=1, target='s0', depth=2)
        visualization_path(model=enzyme_model, path=paths[0], target_node='s0',
                           type_analysis='consumption', filename='test_path_visualization.eps')
