"""Tests for the C++ hdf5_io package."""

import importlib.util
import os
import sys
import tempfile
import types
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

import hdf5_io

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON3_REF = REPO_ROOT / "src" / "python3" / "hdf5_io.py"


def load_reference_python_module():
    """Load the original reference implementation without installing it."""
    spec = importlib.util.spec_from_file_location("hdf5_io_reference", PYTHON3_REF)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def dummy_function():
    pass


class DummyClass(hdf5_io.Hdf5Exportable):
    def __init__(self):
        super().__init__()
        self.data = []

    def dummy_method(self, obj):
        self.data.append(obj)


def gen_example_data():
    data = {
        'None': None,
        'scalars': [0, np.int64(1), 2.0, np.float64(3.0), 4.0j, 'five', True, 2**70,
                    b'a byte string'],
        'arrays': [np.array([6, 66]), np.array([]), np.zeros([]), np.array([True, False])],
        'masked_arrays': [
            np.ma.masked_equal([[1, -1, 3], [-1, 5, 6]], -1),
            np.ma.masked_array(
                [[1, -1, 3], [-1, 5, 6]],
                mask=[[True, True, False], [False] * 3],
                fill_value=-1,
            ),
        ],
        'iterables': [[], [11, 12], tuple([]), tuple([1, 2, 3]), set([]), set([1, 2, 3])],
        'recursive': [0, None, 2, [3, None, 5]],
        'dict_complicated': {0: 1, 'asdf': 2, (1, 2): '3'},
        'exportable': hdf5_io.Hdf5Exportable(),
        'range': range(2, 8, 3),
        'dtypes': [np.dtype('int64'), np.dtype([('a', np.int32, 8), ('b', np.float64, 5)])],
    }
    data['recursive'][3][1] = data['recursive'][1] = data['recursive']
    data['exportable'].some_attr = 'something'
    return data


def assert_equal_data(data_imported, data_expected, max_recursion_depth=10):
    assert isinstance(data_imported, type(data_expected)) or (
        type(data_expected) is bytes and isinstance(data_imported, str)
    )
    if hasattr(data_expected, 'test_sanity'):
        data_imported.test_sanity()
    if isinstance(data_expected, dict):
        assert set(data_imported.keys()) == set(data_expected.keys())
        if max_recursion_depth > 0:
            for ki in data_expected.keys():
                assert_equal_data(data_imported[ki], data_expected[ki], max_recursion_depth - 1)
    elif isinstance(data_expected, (list, tuple)):
        assert len(data_imported) == len(data_expected)
        if max_recursion_depth > 0:
            for vi, ve in zip(data_imported, data_expected):
                assert_equal_data(vi, ve, max_recursion_depth - 1)
    elif isinstance(data_expected, np.ndarray):
        np.testing.assert_array_equal(data_imported, data_expected)
    elif isinstance(data_expected, (int, float, np.int64, np.float64, bool)):
        assert data_imported == data_expected
    elif isinstance(data_expected, range):
        assert tuple(data_imported) == tuple(data_expected)
    elif isinstance(data_expected, types.FunctionType):
        assert data_imported is data_expected


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_export_import_cpp(tmp_path):
    data = gen_example_data()
    dc = DummyClass()
    data.update({
        'global_function': dummy_function,
        'global_class': DummyClass,
        'instance': dc,
        'method': dc.dummy_method,
        'excluded_from_load': np.arange(3.0),
    })
    data_with_ignore = data.copy()
    data_with_ignore['ignore_save'] = hdf5_io.Hdf5Ignored()
    filename = tmp_path / 'test.hdf5'
    with h5py.File(filename, 'w') as f:
        hdf5_io.save_to_hdf5(f, data_with_ignore)
        f['ignore_load'] = 'ignore_during_load'
        f['ignore_load'].attrs[hdf5_io.ATTR_TYPE] = hdf5_io.REPR_IGNORED
    with h5py.File(filename, 'r') as f:
        data_imported = hdf5_io.load_from_hdf5(
            f, ignore_unknown=False, exclude=['/excluded_from_load']
        )
    assert isinstance(data_imported['ignore_load'], hdf5_io.Hdf5Ignored)
    del data_imported['ignore_load']
    assert isinstance(data_imported['excluded_from_load'], hdf5_io.Hdf5Ignored)
    data['excluded_from_load'] = hdf5_io.Hdf5Ignored('/excluded_from_load')
    assert_equal_data(data_imported, data)
    assert len(data['instance'].data) == 0
    data['method'](12345)
    assert len(data['instance'].data) == 1


def gen_cross_compat_data():
    """Subset storable by both reference Python and C++ implementations."""
    data = gen_example_data()
    del data['dtypes']
    del data['exportable']
    return data


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_cross_python_save_cpp_load(tmp_path):
    ref = load_reference_python_module()
    data = gen_cross_compat_data()
    filename = tmp_path / 'py_save_cpp_load.hdf5'
    with h5py.File(filename, 'w') as f:
        ref.save_to_hdf5(f, data)
    with h5py.File(filename, 'r') as f:
        loaded = hdf5_io.load_from_hdf5(f)
    assert_equal_data(loaded, data)


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_cross_cpp_save_python_load(tmp_path):
    ref = load_reference_python_module()
    data = gen_cross_compat_data()
    filename = tmp_path / 'cpp_save_py_load.hdf5'
    with h5py.File(filename, 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
    with h5py.File(filename, 'r') as f:
        loaded = ref.load_from_hdf5(f)
    assert_equal_data(loaded, data)


def test_valid_hdf5_path_component():
    assert hdf5_io.valid_hdf5_path_component('foo')
    assert not hdf5_io.valid_hdf5_path_component('a/b')
    assert not hdf5_io.valid_hdf5_path_component('.')


def test_save_load_pickle_roundtrip(tmp_path):
    data = {'x': [1, 2, 3]}
    path = tmp_path / 'data.pkl'
    hdf5_io.save(data, str(path))
    loaded = hdf5_io.load(str(path))
    assert loaded == data
