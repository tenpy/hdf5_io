"""Tools to save and load data (from TeNPy) to disk.

.. note ::
    This file is maintained in the repository https://github.com/tenpy/hdf5_io.git

The functions :func:`dump` and :func:`load` are convenience functions for saving and loading
quite general python objects (like dictionaries) to/from files, guessing the file type
(and hence protocol for reading/writing) from the file ending.

On top of that, this function provides support for saving python objects to [HDF5]_ files with

.. rubric:: Global module constants used for our HDF5 format

Names of HDF5 attributes:

.. autodata:: ATTR_TYPE
.. autodata:: ATTR_CLASS
.. autodata:: ATTR_MODULE
.. autodata:: ATTR_LEN
.. autodata:: ATTR_FORMAT

Names for the ``ATTR_TYPE`` attribute:

.. autodata:: REPR_HDF5EXPORTABLE
.. autodata:: REPR_NONE
.. autodata:: REPR_LIST
.. autodata:: REPR_SET
.. autodata:: REPR_TUPLE
.. autodata:: REPR_DICT_GENERAL
.. autodata:: REPR_DICT_SIMPLE

.. autodata:: TYPES_FOR_HDF5_DATASETS
"""
# Copyright 2020 TeNPy Developers, GNU GPLv3

import pickle
import gzip
import types
import numpy as np
import importlib

__all__ = [
    'dump', 'load', 'Hdf5FormatError', 'Hdf5ExportError', 'Hdf5ImportError', 'Hdf5Exportable',
    'Hdf5Saver', 'Hdf5Loader', 'dump_to_hdf5', 'load_from_hdf5'
]


def dump(data, filename, mode='w'):
    """Save `data` to file with given `filename`.

    This function guesses the type of the file from the filename ending.
    Supported endings:

    ======== ===============================
    ending   description
    ======== ===============================
    .pkl     Pickle without compression
    -------- -------------------------------
    .pklz    Pickle with gzip compression.
    -------- -------------------------------
    .hdf5    Hdf5 file (using `h5py`).
    ======== ===============================

    Parameters
    ----------
    filename : str
        The name of the file where to save the data.
    mode : str
        File mode for opening the file. ``'w'`` for write (discard existing file),
        ``'a'`` for append (add data to exisiting file).
        See :py:func:`open` for more details.
    """
    filename = str(filename)
    if filename.endswith('.pkl'):
        with open(filename, mode + 'b') as f:
            pickle.dump(data, f)
    elif filename.endswith('.pklz'):
        with gzip.open(filename, mode + 'b') as f:
            pickle.dump(data, f)
    elif filename.endswith('.hdf5'):
        import h5py
        with h5py.File(filename, mode) as f:
            dump_to_hdf5(f, obj)
    else:
        raise ValueError("Don't recognise file ending of " + repr(filename))


def load(filename):
    """Load data from file with given `filename`.

    Guess the type of the file from the filename ending, see :func:`dump` for possible endings.

    Parameters
    ----------
    filename : str
        The name of the file to load.

    Returns
    -------
    data : obj
        The object loaded from the file.
    """
    filename = str(filename)
    if filename.endswith('.pkl'):
        with open(filename, mode) as f:
            data = pickle.load(f, 'rb')
    elif filename.endswith('.pklz'):
        with gzip.open(filename, mode) as f:
            data = pickle.load(f, 'rb')
    elif filename.endswith('.hdf5'):
        import h5py
        with h5py.File(filename, 'r') as f:
            data = load_from_hdf5(f)
    else:
        raise ValueError("Don't recognise file ending of " + repr(filename))
    return data


# =================================================================================
# everything below is for our export/import with our self-definded HDF5 format.
# =================================================================================

#: saved object is instance of a user-defined class following the :class:`Hdf5Exportable` style.
REPR_HDF5EXPORTABLE = np.string_("instance")

REPR_ARRAY = np.string_("array")  #: saved object represents a numpy array
REPR_INT = np.string_("int")  #: saved object represents a (python) int
REPR_FLOAT = np.string_("float")  #: saved object represents a (python) float
REPR_STR = np.string_("str")  #: saved object represents a (python unicode) string
REPR_COMPLEX = np.string_("complex")  #: saved object represents a complex number
REPR_INT64 = np.string_("np.int64")  #: saved object represents a np.int64
REPR_FLOAT64 = np.string_("np.float64")  #: saved object represents a np.float64
REPR_INT32 = np.string_("np.int32")  #: saved object represents a np.int32
REPR_FLOAT32 = np.string_("np.float32")  #: saved object represents a np.float32

REPR_NONE = np.string_("None")  #: saved object is ``None``
REPR_RANGE = np.string_("range")  #: saved object is a range
REPR_LIST = np.string_("list")  #: saved object represents a list
REPR_TUPLE = np.string_("tuple")  #: saved object represents a tuple
REPR_SET = np.string_("set")  #: saved object represents a set
REPR_DICT_GENERAL = np.string_("dict")  #: saved object represents a dict with complicated keys
REPR_DICT_SIMPLE = np.string_("simple_dict")  #: saved object represents a dict with simple keys
REPR_DTYPE = np.string_("dtype")  #: saved object represents a np.dtype

#: tuple of (type, type_repr) which h5py can save as datasets; one entry for each type.
TYPES_FOR_HDF5_DATASETS = tuple([
    (np.ndarray, REPR_ARRAY),
    (int, REPR_INT),
    (float, REPR_FLOAT),
    (str, REPR_STR),
    (complex, REPR_COMPLEX),
    (np.int64, REPR_INT64),
    (np.float64, REPR_FLOAT64),
    (np.int32, REPR_INT32),
    (np.float32, REPR_FLOAT32),
])

ATTR_TYPE = "type"  #: Attribute name for type of the saved object, should be one of the ``REPR_*``
ATTR_CLASS = "class"  #: Attribute name for the class name of an HDF5Exportable
ATTR_MODULE = "module"  #: Attribute name for the module where ATTR_CLASS can be retrieved
ATTR_LEN = "len"  #: Attribute name for the length of iterables, e.g, list, tuple
ATTR_FORMAT = "format"  #: indicates the `ATTR_TYPE` format used by :class:`Hdf5Exportable`


def valid_hdf5_path_component(name):
    """Determine if `name` is a valid HDF5 path component.

    Conditions: String, no ``'/'``, and overall ``name != '.'``.
    """
    # unicode is encoded correctly by h5py and works - amazing!
    return isinstance(name, str) and '/' not in name and name != '.'


class Hdf5FormatError(Exception):
    """Common base class for errors regarding our HDF5 format."""
    pass


class Hdf5ExportError(Hdf5FormatError):
    """This exception is raised when something went wrong during export to hdf5."""
    pass


class Hdf5ImportError(Hdf5FormatError):
    """This exception is raised when something went wrong during import from hdf5."""
    pass


class Hdf5Exportable(object):
    """Interface specification for a class to be exportable to our HDF5 format.

    To allow a class to be exported to HDF5 with :func:`dump_to_hdf5`,
    it only needs to implement the :meth:`save_hdf5` method as documented below.
    To allow import, a class should implement the classmethod :meth:`from_hdf5`.
    During the import, the class already needs to be defined;
    loading can only initialize instances, not define classes.

    The implementation given works for sufficiently simple (sub-)classes, for which all data is
    stored in :attr:`~object.__dict__`.
    In particular, this works for python-defined classes which simply store data using
    ``self.data = data`` in their methods.
    """
    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        """Export `self` into a HDF5 file.

        This method saves all the data it needs to reconstruct `self` with :meth:`from_hdf5`.

        This implementation saves the content of :attr:`~object.__dict__` with
        :meth:`~tenpy.tools.io.Hdf5Saver.save_dict_content`,
        storing the format under the attribute ``'format'``.

        Parameters
        ----------
        hdf5_saver : :class:`~tenpy.tools.io.Hdf5Saver`
            Instance of the saving engine.
        h5gr : :class`Group`
            HDF5 group which is supposed to represent `self`.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.
        """
        # for new implementations, use:
        #   hdf5_saver.dump(data, subpath + "key")  # for big content/data
        #   h5gr.attrs["name"] = info               # for metadata

        # here: assume all the data is given in self.__dict__
        type_repr = hdf5_saver.save_dict_content(self.__dict__, h5gr, subpath)
        h5gr.attrs[ATTR_FORMAT] = type_repr

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        """Load instance from a HDF5 file.

        This method reconstructs a class instance from the data saved with :meth:`save_hdf5`.

        Parameters
        ----------
        hdf5_loader : :class:`~tenpy.tools.io.Hdf5Loader`
            Instance of the loading engine.
        h5gr : :class:`Group`
            HDF5 group which is represent the object to be constructed.
        subpath : str
            The `name` of `h5gr` with a ``'/'`` in the end.

        Returns
        -------
        obj : cls
            Newly generated class instance containing the required data.
        """
        # for new implementations, use:
        #   obj = cls.__new__(cls)                     # create class instance, no __init__() call
        #   hdf5_loader.memorize(h5gr, obj)            # call preferably before loading other data
        #   info = hdf5_loader.get_attr(h5gr, "name")  # for metadata
        #   data = hdf5_loader.load(subpath + "key")   # for big content/data

        dict_format = hdf5_loader.get_attr(h5gr, ATTR_FORMAT)
        obj = cls.__new__(cls)  # create class instance, no __init__() call
        hdf5_loader.memorize(h5gr, obj)  # call preferably before loading other data
        data = hdf5_loader.load_dict(h5gr, dict_format, subpath)  # specialized loading
        # (the `load_dict` did not overwrite the memo entry)
        obj.__dict__.update(data)  # store data in the object
        return obj


class Hdf5Saver:
    """Engine to save simple enough objects into a HDF5 file.

    The intended use of this class is through :func:`dump_to_hdf5`, which is simply an alias
    for ``Hdf5Saver(h5group).dump(obj, path)``.

    It exports python objects to a HDF5 file such that they can be loaded with the
    :class:`Hdf5Loader`, or a call to :func:`load_from_hdf5`, respectively.

    The basic structure of this class is similar as the `Pickler` from :mod:`pickle`.

    See :doc:`/intro/input_output` for a specification of what can be dumped and what the resulting
    datastructure is.

    Parameters
    ----------
    h5group : :class:`Group`
        The HDF5 group (or HDF5 :class:`File`) where to save the data.
    format_selection : dict
        This dictionary allows to set a output format selection for user-defined
        :meth:`Hdf5Exportable.save_hdf5` implementations.
        For example, :class:`~tenpy.linalg.LegCharge` checks it for the key ``"LegCharge"``.

    Attributes
    ----------
    h5group : :class:`Group`
        The HDF5 group (or HDF5 :class:`File`) where to save the data.
    dispatch : dict
        Mapping from a type `keytype` to methods `f` of this class.
        The method is called as ``f(self, obj, path, type_repr)``.
        The call to `f` should save the object `obj` in ``self.h5group[path]``,
        call :meth:`memorize`, and set ``h5gr.attr[ATTR_TYPE] = type_repr``
        to a string `type_repr` in order to allow loading with the dispatcher
        in ``Hdf5Loader.dispatch[type_repr]``.
    memo : dict
        A dictionary to remember all the objects which we already stored to :attr:`h5group`.
        The dictionary keys are object ids; the values are two-tuples of the hdf5 group or dataset
        where an object was stored, and the object itself. See :meth:`memorize`.
    format_selection : dict
        This dictionary allows to set a output format selection for user-defined
        :meth:`Hdf5Exportable.save_hdf5` implementations.
        For example, :class:`~tenpy.linalg.LegCharge` checks it for the key ``"LegCharge"``.
    """
    def __init__(self, h5group, format_selection=None):
        self.h5group = h5group
        self.memo = {}
        if format_selection is None:
            format_selection = {}
        self.format_selection = format_selection

    def dump(self, obj, path='/'):
        """Save `obj` in ``self.h5group[path]``.

        Parameters
        ----------
        obj : object
            The object (=data) to be saved.
        path : str
            Path within `h5group` under which the `obj` should be saved.
            To avoid unwanted overwriting of important data, the group/object should not yet exist,
            except if `path` is the default ``'/'``.

        Returns
        -------
        h5gr : :class:`Group` | :class:`Dataset`
            The h5py group or dataset in which `obj` was saved.
        """
        obj_id = id(obj)
        in_memo = self.memo.get(obj_id)  # default=None
        if in_memo is not None:  # saved the object before
            h5gr, _ = in_memo
            self.h5group[path] = h5gr  # create hdf5 hard link
            # hard linked objects share an hdf5 id,
            # which we use in the loader to distinguish them
            return h5gr

        disp = self.dispatch.get(type(obj))
        if disp is not None:
            f, type_repr = disp
            # `f` is a dispatcher function, which should
            # - save the `obj` in self.h5group['path'],
            # - call :meth:`memorize`, and
            # - set ``h5gr.attr[ATTR_TYPE] = type_repr`` to a string `type_repr`
            #   to allow loading with the dispatcher ``Hdf5Loader.dispatch[type_repr]``
            # call unbound method `f` with explicit self
            h5gr = f(self, obj, path, type_repr)
            return h5gr

        # handle classes with `save_hdf5` method
        obj_save_hdf5 = getattr(obj, 'save_hdf5', None)
        if obj_save_hdf5 is not None:  # of Hdf5Exportable type
            # `obj_save_hdf5` should be the bound method `obj.save_hdf5`,
            # so it does not need an explicit reference of `obj`
            h5gr, subpath = self.create_group_for_obj(path, obj)
            h5gr.attrs[ATTR_TYPE] = REPR_HDF5EXPORTABLE
            h5gr.attrs[ATTR_CLASS] = obj.__class__.__name__  # preferably __qualname,
            # but that doesn't exist in python 2.7
            h5gr.attrs[ATTR_MODULE] = obj.__class__.__module__
            obj_save_hdf5(self, h5gr, subpath)  # should save the actual data
            return h5gr

        # unknown case
        msg = "Don't know how to dump object of type {0!r}".format(type(obj))
        raise Hdf5ExportError(msg)

    def create_group_for_obj(self, path, obj):
        """Create an HDF5 group ``self.h5group[path]`` to store `obj`.

        Also handle ending of path with ``'/'``, and memorize `obj` in the :attr:`memo`.

        Parameters
        ----------
        path : str
            Path within `h5group` under which the `obj` should be saved.
            To avoid unwanted overwriting of important data, the group/object should not yet exist,
            except if `path` is the default ``'/'``.
        obj : object
            The object (=data) to be saved.

        Returns
        -------
        h5group : :class:`Group`
            Newly created h5py (sub)group ``self.h5group[path]``, unless `path` is ``'/'``,
            in which case it is simply the existing ``self.h5group['/']``.
        subpath : str
            The `group.name` ending with ``'/'``, such that other names can be appended to
            get the path for subgroups or datasets in the group.

        Raises
        ------
        ValueError : if `self.h5group[path]`` already existed and `path` is not ``'/'``.
        """
        if path == '/':
            gr = self.h5group[path]
        else:
            gr = self.h5group.create_group(path)  # raises ValueError if path already exists.
        subpath = path if path[-1] == '/' else (path + '/')
        self.memorize(gr, obj)
        return gr, subpath

    def memorize(self, h5gr, obj):
        """Store objects already saved in the :attr:`memo`.

        This allows to avoid copies, if the same python object appears multiple times in the
        data of `obj`. Examples can be shared :class:`~tenpy.linalg.charges.LegCharge` objects
        or even shared :class:`~tenpy.linalg.np_conserved.Array`.
        Using the memo also avoids crashes from cyclic references,
        e.g., when a list contains a reference to itself.

        Parameters
        ----------
        h5gr : :class:`Group` | :class:`Dataset`
            The h5py group or dataset in which `obj` was saved.
        obj : :class:`object`
            The object saved.
        """
        obj_id = id(obj)
        assert obj_id not in self.memo
        self.memo[obj_id] = (h5gr, obj)

    dispatch = {}

    # the methods below are used in the dispatch table

    def save_none(self, obj, path, type_repr):
        """Save the None object as a string (dataset); in dispatch table."""
        self.h5group[path] = REPR_NONE
        h5gr = self.h5group[path]
        h5gr.attrs[ATTR_TYPE] = REPR_NONE
        return h5gr

    dispatch[type(None)] = (save_none, REPR_NONE)

    def save_dataset(self, obj, path, type_repr):
        """Save `obj` as a hdf5 dataset; in dispatch table."""
        self.h5group[path] = obj  # save as dataset
        h5gr = self.h5group[path]
        h5gr.attrs[ATTR_TYPE] = type_repr
        return h5gr

    for _t, _type_repr in TYPES_FOR_HDF5_DATASETS:
        dispatch[_t] = (save_dataset, _type_repr)
    dispatch[unicode] = (save_dataset, REPR_STR)

    def save_string(self, obj, path, type_repr):
        self.h5group[path] = obj.decode()  # save all strings as unicode strings!
        h5gr = self.h5group[path]
        h5gr.attrs[ATTR_TYPE] = type_repr
        return h5gr
    dispatch[str] = (save_string, REPR_STR)

    def save_iterable(self, obj, path, type_repr):
        """Save an iterable `obj` like a list, tuple or set; in dispatch table."""
        h5gr, subpath = self.create_group_for_obj(path, obj)
        h5gr.attrs[ATTR_TYPE] = type_repr
        self.save_iterable_content(obj, h5gr, subpath)
        return h5gr

    dispatch[list] = (save_iterable, REPR_LIST)
    dispatch[tuple] = (save_iterable, REPR_TUPLE)
    dispatch[set] = (save_iterable, REPR_SET)

    def save_iterable_content(self, obj, h5gr, subpath):
        """Save contents of an iterable `obj` in the existing `h5gr`.

        Parameters
        ----------
        obj : dict
            The data to be saved
        h5gr : :class:`Group`
            h5py Group under which the keys and values of `obj` should be saved.
        subpath : str
            Name of h5gr with ``'/'`` in the end.
        """
        h5gr.attrs[ATTR_LEN] = len(obj)
        for i, elem in enumerate(obj):
            self.dump(elem, subpath + str(i))

    def save_dict(self, obj, path, type_repr):
        """Save the dictionary `obj`; in dispatch table."""
        h5gr, subpath = self.create_group_for_obj(path, obj)
        type_repr = self.save_dict_content(obj, h5gr, subpath)
        h5gr.attrs[ATTR_TYPE] = type_repr
        return h5gr

    dispatch[dict] = (save_dict, REPR_DICT_GENERAL)

    def save_dict_content(self, obj, h5gr, subpath):
        """Save contents of a dictionary `obj` in the existing `h5gr`.

        The format depends on whether the dictionary `obj` has simple keys valid for hdf5 path
        components (see :func:`valid_hdf5_path_component`) or not.
        For simple keys: directly use the keys as path.
        For non-simple keys: save list of keys und ``"keys"`` and list of values und ``"values"``.

        Parameters
        ----------
        obj : dict
            The data to be saved
        h5gr : :class:`Group`
            h5py Group under which the keys and values of `obj` should be saved.
        subpath : str
            Name of h5gr with ``'/'`` in the end.

        Returns
        -------
        type_repr : REPR_DICT_SIMPLE | REPR_DICT_GENERAL
            Indicates whether the data was saved in the format for a dictionary with simple keys
            or general keys, see comment above.
        """
        # check if we have only simple keys, which we can use in `path`
        simple_keys = True
        for k in obj.keys():
            if not valid_hdf5_path_component(k):
                simple_keys = False
                break

        if simple_keys:
            for k, v in obj.items():
                self.dump(v, subpath + k)
            return REPR_DICT_SIMPLE
        else:
            keys = obj.keys()
            values = obj.values()
            self.save_iterable(keys, subpath + "keys", REPR_LIST)
            self.save_iterable(values, subpath + "values", REPR_LIST)
            return REPR_DICT_GENERAL

    def save_range(self, obj, path, type_repr):
        """Save a range object; in dispatch table."""
        h5gr, subpath = self.create_group_for_obj(path, obj)
        h5gr.attrs[ATTR_TYPE] = REPR_RANGE
        # there is no direct way to read out the start, stop and step
        # hack: repr() displays 'xrange(...)'
        args = repr(obj)[7:-1].split(',')
        if len(args) == 1:
            start = 0
            stop = int(args[0])
            step = 1
        elif len(args) == 2:
            start, stop = [int(a) for a in args]
            step = 1
        else:
            start, stop, step = [int(a) for a in args]
        self.dump(start, subpath + 'start')
        self.dump(stop, subpath + 'stop')
        self.dump(step, subpath + 'step')
        return h5gr
    dispatch[xrange] = (save_range, REPR_RANGE)

    def save_dtype(self, obj, path, type_repr):
        """Save a :class:`~numpy.dtype` object; in dispatch table."""
        h5gr, subpath = self.create_group_for_obj(path, obj)
        h5gr.attrs[ATTR_TYPE] = REPR_DTYPE
        name = getattr(obj, "name", "void")
        h5gr.attrs["name"] = name
        self.dump(obj.descr, subpath + 'descr')
        return h5gr

    dispatch[np.dtype] = (save_dtype, REPR_DTYPE)

    # clean up temporary variables
    del _t
    del _type_repr


class Hdf5Loader:
    """Class to load and import object from a HDF5 file.

    The intended use of this class is through :func:`load_from_hdf5`, which is simply an alias
    for ``Hdf5Loader(h5group).load(path)``.

    It can load data exported with :func:`dump_to_hdf5` or the :class:`Hdf5Saver`, respectively.

    The basic structure of this class is similar as the `Unpickler` from :mod:`pickle`.

    See :doc:`/intro/input_output` for a specification of what can be dumped and what the resulting
    datastructure is.

    Parameters
    ----------
    h5group : :class:`Group`
        The HDF5 group (or file) where to save the data.

    Attributes
    ----------
    h5group : :class:`Group`
        The HDF5 group (or HDF5 :class:`File`) where to save the data.
    dispatch : dict
        Mapping from a :class:`np.string_`, which is one of the global ``REPR_*`` variables,
        to methods `f` of this class.
        The method is called as ``f(self, h5gr, type_info, subpath)``.
        The call to `f` should load and return an object `obj` from the h5py :class:`Group`
        or :class:`Dataset` `h5gr`; and memorize the loaded `obj` with :meth:`memorize`.
        `subpath` is just the name of `h5gr` with a guaranteed ``'/'`` in the end.
        `type_info` is often the ``REPR_*`` variable of the type or some other information about
        the type, which allows to use a single dispatch function for different datatypes.
    memo : dict
        A dictionary to remember all the objects which we already loaded from :attr:`h5group`.
        The dictionary keys are h5py Group- or dataset ``id``;
        the values are the loaded objects. See :meth:`memorize`.
    """
    def __init__(self, h5group):
        self.h5group = h5group
        self.memo = {}

    def load(self, path=None):
        """Load a Python :class:`object` from the dataset.

        See :func:`load_from_hdf5` for more details.

        Parameters
        ----------
        path : None | str | :class:`Reference`
            Path within :attr:`h5group` to be used for loading.
            Defaults to the name of :attr:`h5group` itself.

        Returns
        -------
        obj : object
            The Python object loaded from `h5group` (specified by `path`).
        """
        # get dataset to be loaded
        if path is None:
            h5gr = self.h5group
            path = self.h5group.name
        else:
            h5gr = self.h5group[path]
        subpath = path if path[-1] == '/' else (path + '/')
        # check memo
        in_memo = self.memo.get(h5gr.id)  # default=None
        if in_memo is not None:  # loaded the object before
            return in_memo

        # determine type of object to be loaded.
        type_repr = np.string_(self.get_attr(h5gr, ATTR_TYPE))
        disp = self.dispatch.get(type_repr)
        if disp is None:
            msg = "Unknown type {0!r} while loading hdf5 dataset {1!s}"
            raise Hdf5ImportError(msg.format(type_repr, h5gr.name))
        f, type_info = disp
        # `f` is a dispatcher function, which should do the following
        # (preferably in this order, if `obj` is mutable):
        # - generate an object `obj` of the described type
        # - call :meth:`memorize` for the generated `obj`,
        # - fill the object with the data from subgroups/subdatasets (everything under `subpath`)
        # - return the generated `obj`
        # call unbound method `f` with explicit self
        obj = f(self, h5gr, type_info, subpath)
        return obj

    def memorize(self, h5gr, obj):
        """Store objects already loaded in the :attr:`memo`.

        This allows to avoid copies, if the same dataset appears multiple times in the
        hdf5 group of `obj`.
        Examples can be shared :class:`~tenpy.linalg.charges.LegCharge` objects
        or even shared :class:`~tenpy.linalg.np_conserved.Array`.

        To handle cyclic references correctly, this function should be called *before* loading
        data from subgroups with new calls of :meth:`load`.
        """
        self.memo.setdefault(h5gr.id, obj)  # don't overwrite existing entries!

    @staticmethod
    def get_attr(h5gr, attr_name):
        """Return attribute ``h5gr.attrs[attr_name]``, if existent.

        Raises
        ------
        :class:`Hdf5ImportError`
            If the attribute does not exist.
        """
        res = h5gr.attrs.get(attr_name)
        if res is None:
            msg = "missing attribute {0!r} for dataset {1!s}"
            raise Hdf5ImportError(msg.format(attr_name, h5gr.name))
        return res

    @staticmethod
    def find_class(module, classname):
        """Get the class of the qualified `classname` in a given python `module`.

        Imports the module."""
        mod = importlib.import_module(module)
        cls = mod
        for subpath in classname.split('.'):
            cls = getattr(cls, subpath)
        return cls

    dispatch = {}

    # the methods below are used in the dispatch table

    def load_none(self, h5gr, type_info, subpath):
        """Load the ``None`` object from a dataset."""
        obj = None
        self.memorize(h5gr, obj)
        return obj

    dispatch[REPR_NONE] = (load_none, None)

    def load_dataset(self, h5gr, type_info, subpath):
        """Load a h5py :class:`Dataset` and convert it into the desired type."""
        if type_info is np.ndarray:
            obj = h5gr[...]
        else:
            obj = h5gr[()]  # load scalar from hdf5 Dataset
            # convert to desired type: type_info is simply the type
            obj = type_info(obj)
        self.memorize(h5gr, obj)
        return obj

    for _t, _type_repr in TYPES_FOR_HDF5_DATASETS:
        dispatch[_type_repr] = (load_dataset, _t)

    def load_list(self, h5gr, type_info, subpath):
        """Load a list."""
        obj = []
        self.memorize(h5gr, obj)
        length = self.get_attr(h5gr, ATTR_LEN)
        for i in range(length):
            sub_obj = self.load(subpath + str(i))
            obj.append(sub_obj)
        return obj

    dispatch[REPR_LIST] = (load_list, REPR_LIST)

    def load_set(self, h5gr, type_info, subpath):
        """Load a set."""
        obj = set([])
        self.memorize(h5gr, obj)
        length = self.get_attr(h5gr, ATTR_LEN)
        for i in range(length):
            sub_obj = self.load(subpath + str(i))
            obj.add(sub_obj)
        return obj

    dispatch[REPR_SET] = (load_set, REPR_SET)

    def load_tuple(self, h5gr, type_info, subpath):
        """Load a tuple."""
        obj = []  # tuple is immutable: can't append to it
        # so we need to use a list during loading
        self.memorize(h5gr, obj)
        # BUG: for recursive tuples, the memorized object is a list instead of a tuple.
        # but I don't know how to circumvent this.
        # It's hopefully not relevant for our applications.
        length = self.get_attr(h5gr, ATTR_LEN)
        for i in range(length):
            sub_obj = self.load(subpath + str(i))
            obj.append(sub_obj)
        # now conjvert the list to tuple
        obj = tuple(obj)
        self.memo[h5gr.id] = obj  # overwrite the memo entry to point to the tuple, not the list
        return obj

    dispatch[REPR_TUPLE] = (load_tuple, REPR_TUPLE)

    def load_dict(self, h5gr, type_info, subpath):
        """Load a dictionary in the format according to `type_info`."""
        if type_info == REPR_DICT_GENERAL:
            return self.load_general_dict(h5gr, type_info, subpath)
        elif type_info == REPR_DICT_SIMPLE:
            return self.load_simple_dict(h5gr, type_info, subpath)
        raise ValueError("can't interpret type_info {0!r}".format(type_info))

    def load_general_dict(self, h5gr, type_info, subpath):
        """Load a dictionary with general keys."""
        obj = {}
        self.memorize(h5gr, obj)
        keys = self.load_list(h5gr['keys'], REPR_LIST, subpath + 'keys/')
        values = self.load_list(h5gr['values'], REPR_LIST, subpath + 'values/')
        obj.update(zip(keys, values))
        return obj

    dispatch[REPR_DICT_GENERAL] = (load_general_dict, REPR_DICT_GENERAL)

    def load_simple_dict(self, h5gr, type_info, subpath):
        """Load a dictionary with simple keys."""
        obj = {}
        self.memorize(h5gr, obj)
        for k in h5gr.keys():
            v = self.load(subpath + k)
            obj[k] = v
        return obj

    dispatch[REPR_DICT_SIMPLE] = (load_simple_dict, REPR_DICT_SIMPLE)

    def load_range(self, h5gr, type_info, subpath):
        """Load a range."""
        start = self.load(subpath + 'start')
        stop = self.load(subpath + 'stop')
        step = self.load(subpath + 'step')
        obj = xrange(start, stop, step)
        self.memorize(h5gr, obj)  # late, but subgroups should only be int's; no cyclic reference
        return obj

    dispatch[REPR_RANGE] = (load_range, REPR_RANGE)

    def load_dtype(self, h5gr, type_info, subpath):
        """Load a :class:`numpy.dtype`."""
        name = self.get_attr(h5gr, "name")
        if name.startswith("void"):
            descr = self.load(subpath + 'descr')
            obj = np.dtype(descr)
        else:
            obj = np.dtype(name)
        self.memorize(h5gr, obj)
        return obj

    dispatch[REPR_DTYPE] = (load_dtype, REPR_DTYPE)

    def load_hdf5exportable(self, h5gr, type_info, subpath):
        """Load an instance of a userdefined class."""
        modulename = self.get_attr(h5gr, ATTR_MODULE)
        classname = self.get_attr(h5gr, ATTR_CLASS)
        cls = self.find_class(modulename, classname)
        return cls.from_hdf5(self, h5gr, subpath)

    dispatch[REPR_HDF5EXPORTABLE] = (load_hdf5exportable, REPR_HDF5EXPORTABLE)

    # clean up temporary variables
    del _t
    del _type_repr


def dump_to_hdf5(h5group, obj, path='/'):
    """Save an object `obj` into a hdf5 file or group.

    Roughly equivalent to ``h5group[path] = obj``, but handle different types of `obj`.
    For example, dictionaries are handled recursively.
    See :doc:`/intro/input_output` for a specification of what can be dumped and what the resulting
    datastructure is.

    Parameters
    ----------
    h5group : :class:`Group`
        The HDF5 group (or h5py :class:`File`) to which `obj` should be saved.
    obj : object
        The object (=data) to be saved.
    path : str
        Path within `h5group` under which the `obj` should be saved.
        To avoid unwanted overwriting of important data, the group/object should not yet exist,
        except if `path` is the default ``'/'``.

    Returns
    -------
    h5obj : :class:`Group` | :class:`Dataset`
        The h5py group or dataset under which `obj` was saved.
    """
    return Hdf5Saver(h5group).dump(obj, path)


def load_from_hdf5(h5group, path=None):
    """Load an object from hdf5 file or group.

    Roughly equivalent to ``obj = h5group[path][...]``, but handle more complicated objects saved
    as hdf5 groups and/or datasets with :func:`dump_to_hdf5`.
    For example, dictionaries are handled recursively.
    See :doc:`/intro/input_output` for a specification of what can be dumped/loaded and what the
    corresponding datastructure is.

    Parameters
    ----------
    h5group : :class:`Group`
        The HDF5 group (or h5py :class:`File`) to be loaded.
    path : None | str | :class:`Reference`
        Path within `h5group` to be used for loading. Defaults to the `h5group` itself specified.

    Returns
    -------
    obj : object
        The Python object loaded from `h5group` (specified by `path`).
    """
    return Hdf5Loader(h5group).load(path)
