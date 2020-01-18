"""Test output to and import from hdf5."""

import os
import pytest
import warnings
import tempfile
import hdf5_io  # the local version
import numpy as np
import h5py

datadir = os.path.join(os.path.dirname(__file__), 'data')
datadir_files = []
if os.path.isdir(datadir):
    datadir_files = os.listdir(datadir)
datadir_hdf5 = [f for f in datadir_files if f.endswith('.hdf5')]


def gen_example_data():
    data = {
        'None': None,
        'scalars': [0, np.int64(1), 2., np.float64(3.), 4.j, 'five'],
        'arrays': [np.array([6, 66]), np.array([]),
                    np.zeros([])],
        'iterables': [[], [11, 12],
                        tuple([]),
                        tuple([1, 2, 3]),
                        set([]),
                        set([1, 2, 3])],
        'recursive': [0, None, 2, [3, None, 5]],
        'dict_complicated': {
            0: 1,
            'asdf': 2,
            (1, 2): '3'
        },
        'exportable': hdf5_io.Hdf5Exportable(),
        'range': range(2, 8, 3),
        'dtypes': [np.dtype("int64"),
                   np.dtype([('a', np.int32, 8), ('b', np.float64, 5)])],
    }
    data['recursive'][3][1] = data['recursive'][1] = data['recursive']
    data['exportable'].some_attr = "something"
    return data


def assert_equal_data(data_imported, data_expected, max_recursion_depth=10):
    """Check that the imported data is as expected."""
    assert isinstance(data_imported, type(data_expected))
    if hasattr(data_expected, 'test_sanity'):
        data_imported.test_sanity()
    if isinstance(data_expected, dict):
        assert set(data_imported.keys()) == set(data_expected.keys())
        if max_recursion_depth > 0:
            for ki in data_imported.keys():
                assert_equal_data(data_imported[ki], data_expected[ki], max_recursion_depth - 1)
    elif isinstance(data_expected, (list, tuple)):
        if max_recursion_depth > 0:
            for vi, ve in zip(data_imported, data_expected):
                assert_equal_data(vi, ve, max_recursion_depth - 1)
    elif isinstance(data_expected, np.ndarray):
        np.testing.assert_array_equal(data_imported, data_expected)
    elif isinstance(data_expected, (int, float, np.int64, np.float64)):
        assert data_imported == data_expected
    elif isinstance(data_expected, range):
        assert tuple(data_imported) == tuple(data_expected)


def export_to_datadir():
    filename = "data/exported_python3.hdf5"
    data = gen_example_data()
    with h5py.File(filename, 'w') as f:
        hdf5_io.dump_to_hdf5(f, data)


def test_hdf5_export_import():
    """Try subsequent export and import to pickle."""
    data = gen_example_data()
    with tempfile.TemporaryDirectory() as tdir:
        filename = 'test.hdf5'
        with h5py.File(os.path.join(tdir, filename), 'w') as f:
            hdf5_io.dump_to_hdf5(f, data)
        with h5py.File(os.path.join(tdir, filename), 'r') as f:
            data_imported = hdf5_io.load_from_hdf5(f)
    assert_equal_data(data_imported, data)


@pytest.mark.parametrize('fn', datadir_hdf5)
def test_import_from_datadir(fn):
    print("import ", fn)
    filename = os.path.join(datadir, fn)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with h5py.File(filename, 'r') as f:
            data = hdf5_io.load_from_hdf5(f)
    data_expected = gen_example_data()
    assert_equal_data(data, data_expected)


if __name__ == "__main__":
    export_to_datadir()
