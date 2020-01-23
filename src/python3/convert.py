#!/usr/bin/env python3
"""Convert the format of hdf5 files.

Specify from which format into which format you want to convert with the ``--from-format``
and ``--to-format`` command line arguments. We try to load the corresponding conversion
from the file ``convert_from_{FROM_FORMAT}_to_{TO_FORMAT}.py``.
"""
# Try command line argument ``--help`` for options.

import argparse

import h5py
import hdf5_io
from hdf5_io import ATTR_TYPE, ATTR_CLASS, ATTR_MODULE, REPR_HDF5EXPORTABLE, REPR_IGNORED
import importlib


BACKUP_PATH = "/backup_pre_convert"   # path of (temporary) group inside the file
# to move groups to be converted into.


__all__ = ['BACKUP_PATH', 'convert_file', 'parse_args', 'main']


class Hdf5Converter(hdf5_io.Hdf5Loader, hdf5_io.Hdf5Saver):
    """Hdf5Converter base class.

    Prototype for a converted, needs to be subclasses for specific conversion from
    one format into another.
    To make :func:`find_converter` work, subclasses should be implemented in separate files called
    ``'convert_from_{library1}_to_{library2}.py'``.

    Parameters
    ----------
    filename : str
        The name of the hdf5 file to open.
    no_backup : bool
        Whether to keep (False, default) or remove (True) the backups of the converted groups.
    verbose : int
        Verbosity level, i.e., how much to print about what the converter does.

    Attributes
    ----------
    no_backup : bool
        See above.
    verbose : int
        See above
    memo_convert : dict
        A dictionary to remember all the HDF5 groups which we already converted.
        The dictionary key is the h5py group- or dataset ``id``;
        the value is the hdf5 group itself. See :meth:`memorize_convert`.
    mappings : dict
        Dictionary keys are tuples ``(modulename, classname)`` defining objects of which classes
        should be converted, dictionary values are corresponding tuples
        ``(modulename, classname, map_function)`` defining into which class they should be
        converted, with the `map_function` performing the data
        conversion. The `map_function` gets called by :meth:`convert_group` as
        ``map_function(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new)``
        with the old hdf5 group `h5gr_old` moved into :attr:`backup_gr`,  the new group `h5gr_new`
        to take the the converted data, and `subpath_orig` and `subpath_new` just being the names
        of the group with a ``'/'`` in the end to allow easy loading/saving of group members.
        The attributes for type, class and module set accordingly to `h5gr_new`;
        the `map_function` only needs to convert the data.
    backup_gr : :class:`Group`
        Hdf5 group into which groups are moved for conversion.
    """

    def __init__(self, filename, no_backup=False, verbose=0):
        if not (filename.endswith('.h5') or filename.endswith('.hdf5')):
            raise ValueError("unexpected filename ending")
        h5file = h5py.File(filename, 'r+')
        hdf5_io.Hdf5Loader.__init__(self, h5file)
        hdf5_io.Hdf5Saver.__init__(self, h5file)
        self.no_backup = no_backup
        if not hasattr(self, 'mappings'):
            raise NotImplementedError("subclass didn't define self.mappings")
        self.memo_convert = {}
        self.verbose = verbose

        if BACKUP_PATH not in self.h5group:
            self.backup_gr = self.h5group.create_group(BACKUP_PATH)
            self.backup_gr.attrs[ATTR_TYPE] = REPR_IGNORED
        else:
            self.backup_gr = self.h5group[BACKUP_PATH]
        self.memorize_convert(self.backup_gr)  # exclude backup path for conversion


    def memorize_convert(self, h5gr):
        """Store groups already converted in the :attr:`memo_convert`.

        Necessary to avoid loops in the recursion for cyclic references, and to avoid converting
        the same group twice if hardlinks are used.

        Parameters
        ----------
        h5gr : :class:`Group` | :class:`Dataset`
            The h5py group or dataset before conversion.
        """
        self.memo_convert[h5gr.id] = h5gr

    def convert_file(self):
        """Convert all groups of the file."""
        if self.verbose:
            if self.verbose > 1:
                print("="*80)
            print("converting file ", self.h5group.filename)

        self.convert_group(self.h5group)  # recursive

        # now (once that everything worked, not in __del__) we can delete the backup
        if self.no_backup:
            del self.h5group[self.backup_gr.name]

    def convert_group(self, h5gr):
        """Convert a given group, and recursively any subgroups.

        Only groups with the ``'type'`` attribute being REPR_HDF5EXPORTABLE get converted,
        if there is a mapping for the corresponding module and class name combination defined
        in :attr:`mappings`. The latter should be defined by subclasses, see the class doc-string.

        Parameters
        ----------
        h5gr : :class:`Group` | :class:`Dataset`
            The h5py group or dataset to be converted.
        """
        in_memo = self.memo_convert.get(h5gr.id)
        if in_memo is not None:
            return  # already converted
        self.memorize_convert(h5gr)

        type_ = h5gr.attrs.get(ATTR_TYPE)
        if type_ != REPR_HDF5EXPORTABLE:
            if type_ != REPR_IGNORED:
                self.convert_subgroups(h5gr)
            return  # nothing to convert
        module_name = self.get_attr(h5gr, ATTR_MODULE)
        class_name = self.get_attr(h5gr, ATTR_CLASS)
        mapping = self.mappings.get((module_name, class_name))
        if mapping is None:
            self.convert_subgroups(h5gr)
            return  # nothing else to convert

        new_module_name, new_class_name, map_function = mapping
        orig_path = h5gr.name
        moved_path = BACKUP_PATH + h5gr.name
        if self.verbose > 1:
            print("converting group {0!r} from {1!s} to {2!s}"
                  .format(h5gr.name, class_name, new_class_name))
        elif self.verbose:
            print("converting group", h5gr.name)
        self.h5group.move(orig_path, moved_path)  # first move to backup
        h5gr_new, subpath = self.create_group_for_obj(orig_path, h5gr)  # new group for the data
        h5gr_new.attrs[ATTR_TYPE] = REPR_HDF5EXPORTABLE
        h5gr_new.attrs[ATTR_MODULE] = new_module_name
        h5gr_new.attrs[ATTR_CLASS] = new_class_name
        subpath = h5gr.name + '/'
        subpath_new = h5gr_new.name + '/'
        map_function(self, h5gr, subpath, h5gr_new, subpath_new)  # convert data

    def convert_subgroups(self, h5gr):
        """Call :meth:`convert_group` for any subgroups of `h5gr`."""
        if not isinstance(h5gr, h5py.Group):
            return  # nothing to do, in particular for h5py.Dataset
        subgroups = list(h5gr.values())
        for subgr in subgroups:
            self.convert_group(subgr)
        # done

    def recover_from_backup(self):
        """(Try to) recover the backup from :attr:`backup_gr`."""
        if self.verbose:
            print("recovering from backup: ", self.h5group.filename)
        raise NotImplementedError("TODO")  # TODO

    def __del__(self):
        """Close the file properly when the Converter is deleted."""
        h5file = getattr(self, 'h5group', None)
        if h5file is not None:
            h5file.close()


def parse_args(converter_cls=None):
    doc = converter_cls.__module__.__doc__ if converter_cls is not None else __doc__
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument('-B',
                        '--no-backup',
                        action="store_false",
                        help="Don't keep copy of the converted Hdf5 groups inside the file "
                        "under the path {0!r}".format(BACKUP_PATH))
    parser.add_argument('-r',
                        '--recover',
                        action="store_true",
                        help="Restore the groups from the backup under {0!r}".format(BACKUP_PATH))
    parser.add_argument('-v',
                        '--verbose',
                        default=0,
                        action="count",
                        help="Print what we do; give multiple times for more status prints")
    if converter_cls is None:
        parser.add_argument('-f',
                            '--from-format',
                            required=True,
                            help="From which format to convert")
        parser.add_argument('-t',
                            '--to-format',
                            required=True,
                            help="Into which format to convert")
    parser.add_argument("files", nargs="+", help="Filenames of HDF5 files to convert (in place).")
    args = parser.parse_args()
    if converter_cls is not None:
        args.Converter = converter_cls
    return args


def find_converter(from_format, to_format):
    module_name = "convert_from_{0!s}_to_{1!s}".format(from_format, to_format)
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        raise ValueError("Conversion for these formats not possible:\n"
                         "Can't find module {0!r}".format(module_name))
    return mod.Converter


def main(args):
    if 'Converter' in args:
        Converter = args.Converter
    else:
        Converter = find_converter(args.from_format, args.to_format)
    for fn in args.files:
        conv = Converter(fn, args.no_backup, args.verbose)
        if args.recover:
            conv.recover_from_backup()
        else:
            conv.convert_file()
    # done

if __name__ == "__main__":
    args = parse_args()
    main(args)
