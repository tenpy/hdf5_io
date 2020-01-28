#!/usr/bin/env python3
"""Convert the format of hdf5 files from the TeNPy precursor to the new TeNPy."""
# Try command line argument ``--help`` for options.

import convert
from convert import Hdf5Converter

import tenpy  # requirement: can import the new tenpy
import tenpy.linalg.np_conserved as npc
import numpy as np


class Converter(Hdf5Converter):
    """Convert from the `prev_tenpy` format to the new TeNPy."""

    def convert_array(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        # compare algorithms.linalg.np_conserved.array.from_hdf5
        rank = h5gr_orig.attrs["rank"]  # (directly copy the attribute)
        shape = h5gr_orig.attrs["shape"]
        num_q = self.get_attr(h5gr_orig, "num_charges")
        mod_q = self.get_attr(h5gr_orig, "U1_ZN")
        total_charge = self.get_attr(h5gr_orig, "total_charge")
        leg_charges = self.load(subpath_orig + "leg_charges")
        qconj = self.get_attr(h5gr_orig, "qconj")
        blocks = self.load(subpath_orig + "blocks")
        block_inds = np.asarray(self.load(subpath_orig + "block_inds"), np.intp)
        dtype = self.load(subpath_orig + "dtype")
        block_inds_sorted = h5gr_orig.attrs["block_inds_sorted"]
        labels = self.load(subpath_orig + "labels")

        # convert the data as necessary
        # ChargeInfo
        chinfo = npc.ChargeInfo(mod_q)  # no names available
        assert chinfo.qnumber == num_q
        # legs: convert leg_charges, qconj into list of LegCharge instances
        legs_new = []
        for q_ind, q_conj in zip(leg_charges, qconj):
            slices = np.concatenate((q_ind[:, 0], [q_ind[-1, 1]]))
            charges = q_ind[:, 2:]
            leg = npc.LegCharge(chinfo, slices, charges, q_conj)
            legs_new.append(leg)
        # labels: convert {str: int} -> [str]
        labels_new = [None] * rank
        if labels is not None:
            for k, v in labels.items():
                labels_new[v] = k

        # compare tenpy.linalg.np_conserved.Array.save_hdf5
        self.save(chinfo, subpath_new + "chinfo")
        self.save(legs_new, subpath_new + "legs")
        self.save(dtype, subpath_new + "dtype")
        self.save(total_charge, subpath_new + "total_charge")
        self.save(labels_new, subpath_new + "labels")
        self.save(blocks, subpath_new + "blocks")
        self.save(block_inds, subpath_new + "block_inds")
        h5gr_new.attrs["block_inds_sorted"] = block_inds_sorted
        h5gr_new.attrs["rank"] = rank  # not needed for loading, but still usefull metadata
        h5gr_new.attrs["shape"] = shape  # same

    mappings = {}
    mappings[('tools.hdf5_io', 'Hdf5Exportable')] = \
        ('tenpy.tools.hdf5_io', 'Hdf5Exportable', Hdf5Converter.convert_Hdf5Exportable)
    mappings[('algorithms.linalg.np_conserved', 'array')] = \
        ('tenpy.linalg.np_conserved', 'Array', convert_array)


if __name__ == "__main__":
    args = convert.parse_args(converter_cls=Converter)
    convert.main(args)
