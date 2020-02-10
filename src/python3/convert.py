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
import warnings


BACKUP_PATH = "/backup_pre_convert"   #: path of (temporary) group inside the file
# to move groups to be converted into.

ATTR_ORIG_PATH = "original_path"  #: Attribute name for path before moving a group to BACKUP_PATH
# for conversion


__all__ = ['BACKUP_PATH', 'ATTR_ORIG_PATH', 'HDF5Converter', 'parse_args', 'find_converter',
           'main']


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
        the value is the converted hdf5 group. See :meth:`memorize_convert`.
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
        The attributes for type, class and module are set accordingly to `h5gr_new`;
        the `map_function` only needs to convert the data.
        You can use :meth:`load` and :meth:`save` for the conversion, but make (object) copies if
        you want to modify an object, i.e., don't modify the objects "in place".
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
            self.backup_gr = backup_gr = self.h5group[BACKUP_PATH]
            for subgr in backup_gr.values():
                self.memorize_convert(subgr, subgr)
        self.memorize_convert(self.backup_gr, self.backup_gr)  # exclude conversion

    def memorize_convert(self, h5gr, h5gr_converted):
        """Store groups already converted in the :attr:`memo_convert`.

        Necessary to avoid loops in the recursion for cyclic references, and to avoid converting
        the same group twice if hardlinks are used.

        Parameters
        ----------
        h5gr : :class:`Group` | :class:`Dataset`
            The h5py group or dataset before conversion.
        h5gr_converted : :class:`Group` | :class:`Dataset`
            The h5py group or dataset after conversion.
        """
        self.memo_convert[h5gr.id] = h5gr_converted

    def convert_file(self):
        """Convert all groups of the file."""
        if self.verbose:
            if self.verbose > 1:
                print("="*80)
            print("converting file", self.h5group.filename)

        # set top-level group "/" of the file to be a dictionary, if not specified before.
        if not hdf5_io.ATTR_TYPE in self.h5group.attrs:
            self.h5group.attrs[hdf5_io.ATTR_TYPE] = hdf5_io.REPR_DICT_SIMPLE

        self.convert_group(self.h5group)  # recursive

        # now (once that everything worked, not in __del__) we can delete the backup
        if self.no_backup:
            if self.verbose > 1:
                print("remove backup group", self.backup_gr.name)
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

        Returns
        -------
        h5gr_new : :class:`Group` | :class:`Dataset`
            The converted h5py group or dataset.
        """
        in_memo = self.memo_convert.get(h5gr.id)
        if in_memo is not None:
            return in_memo  # already converted
        self.memorize_convert(h5gr, h5gr)  # avoid endless recursion for cyclic references

        type_ = h5gr.attrs.get(ATTR_TYPE)
        if type_ is None:  # no type_ defined
            warnings.warn("Ignore group with no type set.")
            if self.verbose > 1:
                print("dataset {0!r} has no attribute {1!r} defined".format(h5gr.name, ATTR_TYPE))
            return h5gr  # nothing to convert
        if type_ != REPR_HDF5EXPORTABLE:
            if type_ != REPR_IGNORED:
                self.convert_subgroups(h5gr)
            return h5gr  # nothing to convert
        module_name = self.get_attr(h5gr, ATTR_MODULE)
        class_name = self.get_attr(h5gr, ATTR_CLASS)
        mapping = self.mappings.get((module_name, class_name))
        if mapping is None:
            converted_into = False
            for (mod_name, cls_name, _) in self.mappings.values():
                if (mod_name, cls_name) == (module_name, class_name):
                    converted_into = True
                    break
            if not converted_into:
                warnings.warn("no mapping for class {1!r} in {0!r}, simply convert subgroups"
                              .format(module_name, class_name))
                self.convert_subgroups(h5gr)
            return h5gr  # nothing else to convert

        new_module_name, new_class_name, map_function = mapping
        orig_path = h5gr.name
        counter = len(self.backup_gr)
        moved_path =  '/'.join([BACKUP_PATH, str(counter)])
        while moved_path in h5gr:
            counter += 1
            moved_path = '/'.join([BACKUP_PATH, str(counter)])
        if self.verbose > 1:
            print("converting group {0!r} from {1!s} to {2!s}, backup {3!s}"
                  .format(h5gr.name, class_name, new_class_name, moved_path))
        elif self.verbose:
            print("converting group", h5gr.name)
        self.h5group.move(orig_path, moved_path)  # first move to backup
        h5gr.attrs[ATTR_ORIG_PATH] = orig_path
        h5gr_new, subpath = self.create_group_for_obj(orig_path, h5gr)  # new group for the data
        h5gr_new.attrs[ATTR_TYPE] = REPR_HDF5EXPORTABLE
        h5gr_new.attrs[ATTR_MODULE] = new_module_name
        h5gr_new.attrs[ATTR_CLASS] = new_class_name
        subpath = h5gr.name + '/'
        subpath_new = h5gr_new.name + '/'
        self.memorize_convert(h5gr, h5gr_new)  # update memo
        map_function(self, h5gr, subpath, h5gr_new, subpath_new)  # convert data
        return h5gr_new

    def convert_subgroups(self, h5gr):
        """Call :meth:`convert_group` for any subgroups of `h5gr`."""
        if not isinstance(h5gr, h5py.Group):
            return  # nothing to do, in particular for h5py.Dataset
        for subgr in list(h5gr.values()):
            self.convert_group(subgr)
        # done

    def recover_from_backup(self):
        """(Try to) recover the backup from :attr:`backup_gr`."""
        h5file = self.h5group
        backup_gr = self.backup_gr
        if self.verbose:
            print("recovering groups in file {0!r} from backup under {1!r}"
                  .format(h5file.filename, backup_gr.name))
        names = sorted(backup_gr.keys(), reverse=True)  # reverse: handle converted subgroups
        for name in names:
            gr = backup_gr[name]
            orig_path = gr.attrs[ATTR_ORIG_PATH]
            if self.verbose:
                print("recover original {0!r} from group {1!r}".format(orig_path, name))
            if orig_path not in h5file:
                warnings.warn("Expected a converted group under " + orig_path)
            else:
                del h5file[orig_path]
            h5file.move(gr.name, orig_path)
            del gr.attrs[ATTR_ORIG_PATH]

    def __del__(self):
        """Close the file properly when the Converter is deleted."""
        h5file = getattr(self, 'h5group', None)
        if h5file is not None:
            h5file.close()

    @staticmethod  # don't bind to self: allow subclasses to put it in `mappings`.
    def convert_Hdf5Exportable(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        """Convert a group representing an `Hdf5Exportable`."""
        # copy attributes except module, classname and type
        excl_attr_copies = set([ATTR_MODULE, ATTR_CLASS, ATTR_TYPE])
        for attr_name in set(h5gr_orig.attrs.keys()) - excl_attr_copies:
            h5gr_new.attrs[attr_name] = h5gr_orig.attrs[attr_name]
        # copy data sets
        for subgr in h5gr_orig:
            h5gr_new[subgr] = h5gr_orig[subgr]
        # and convert the subgroups
        self.convert_subgroups(h5gr_new)

    @classmethod
    def print_conversions(cls):
        """List from which class into which class conversions are done."""
        print(cls.__doc__)
        line = "{0:40}.{1:20} -> {2:40}.{3:20}"
        print(line.format("from_module", "class", "into_module", "into_class"))
        print("="*120)
        for from_, to_ in cls.mappings.items():
            print(line.format(from_[0], from_[1], to_[0], to_[1]))


def parse_args(converter_cls=None):
    doc = converter_cls.__module__.__doc__ if converter_cls is not None else __doc__
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument('-B',
                        '--no-backup',
                        action="store_true",  # default=False
                        help="Don't keep copy of the converted Hdf5 groups inside the file "
                        "under the path {0!r}".format(BACKUP_PATH))
    parser.add_argument('-r',
                        '--recover',
                        action="store_true",
                        help="Restore the groups from the backup under {0!r} "
                        "instead of converting something".format(BACKUP_PATH))
    parser.add_argument('-l',
                        '--list-only',
                        action="store_true",
                        help="Don't convert anything, just list the classes which get converted.")
    parser.add_argument('-v',
                        '--verbose',
                        default=0,
                        action="count",
                        help="Print what we do; give multiple times for more status prints")
    if converter_cls is None:
        parser.add_argument('-f',
                            '--from-format',
                            help="From which format to convert")
        parser.add_argument('-t',
                            '--to-format',
                            help="Into which format to convert")
    parser.add_argument("files", nargs="*", help="Filenames of HDF5 files to convert (in place).")
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
                         "Can't find module {0!r}".format(module_name)) from None
    return mod.Converter


def main(args):
    if 'Converter' in args:
        Converter = args.Converter
    else:
        Converter = find_converter(args.from_format, args.to_format)
    if args.list_only:
        Converter.print_conversions()
        return
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
