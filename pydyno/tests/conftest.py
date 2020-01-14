import pytest


# in conftest.py
@pytest.fixture(scope='session')
def data_files_dir(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    return datadir
