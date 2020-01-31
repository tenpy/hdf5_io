#!/usr/bin/env python3
# Try command line argument ``--help`` for options.

import convert
from convert import Hdf5Converter
import hdf5_io

import tenpy  # requirement: can import the new tenpy
import tenpy.linalg.np_conserved as npc
import numpy as np


class Converter(Hdf5Converter):
    """Convert from the "new" TeNPy into the "old" TeNpy (in python 2)."""

    mappings = {}
    mappings[('tenpy.tools.hdf5_io', 'Hdf5Exportable')] = \
        ('tools.hdf5_io', 'Hdf5Exportable', Hdf5Converter.convert_Hdf5Exportable)

    def convert_array(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        # copy (hardlinks avoid copies)
        h5gr_new["dtype"] = h5gr_orig["dtype"]
        h5gr_new["blocks"] = h5gr_orig["blocks"]
        total_charge = self.load(subpath_orig + "total_charge")
        h5gr_new.attrs["total_charge"] = total_charge

        block_inds = np.asarray(self.load(subpath_orig + "block_inds"), np.uint)
        self.save(block_inds, subpath_new + "block_inds")
        h5gr_new.attrs["block_inds_sorted"] = h5gr_orig.attrs["block_inds_sorted"]
        h5gr_new.attrs["rank"] = rank = self.get_attr(h5gr_orig, "rank")
        h5gr_new.attrs["shape"] = self.get_attr(h5gr_orig, "shape")

        # convert charges
        chinfo = self.load(subpath_orig + "chinfo")
        legs = self.load(subpath_orig + "legs")
        h5gr_new.attrs["num_charges"] = qnumber = int(chinfo.qnumber)
        h5gr_new.attrs["U1_ZN"] = np.asarray(chinfo.mod, int)
        qconj = np.array([leg.qconj for leg in legs], int)
        h5gr_new.attrs["qconj"] = qconj
        leg_charges = []
        for leg in legs:
            q_ind = np.zeros((leg.block_number, 2 + qnumber), int)
            q_ind[:, 0] = leg.slices[:-1]
            q_ind[:, 1] = leg.slices[1:]
            q_ind[:, 2:] = leg.charges
            leg_charges.append(q_ind)
        self.save(leg_charges, subpath_new + "leg_charges")

        labels = self.load(subpath_orig + "labels")
        # convert [str] => {str: int}
        labels_new = {}
        for k, v in enumerate(labels):
            if v is not None:
                labels_new[v] = k
        self.save(labels_new, subpath_new + "labels")

    mappings[('tenpy.linalg.np_conserved', 'Array')] = \
        ('algorithms.linalg.np_conserved', 'array', convert_array)

    def convert_bc(self, bc):
        """MPS/MPO boundary conditions."""
        # new tenpy:      'finite', 'segment', 'infinite'
        # old tenpy: bc = 'finite', 'segment', 'periodic'
        if bc not in ['finite', 'segment', 'infinite']:
            raise ValueError("unexpected boundary conditions bc={0!r}".format(bc))
        if bc == 'infinite':
            bc = 'periodic'
        return bc

    def convert_MPS(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = self.get_attr(h5gr_orig, "L")

        # tensors
        tensors = self.load(subpath_orig + "tensors")
        # convert leg labels and order
        tensors = [B.itranspose(['p', 'vL', 'vR']).iset_leg_labels(['p', 'b', 'b*'])
                   for B in tensors]
        self.save(tensors, subpath_new + "tensors")
        self.convert_group(h5gr_new["tensors"])  # implicitly calls self.convert_array

        # boundary conditions
        bc = self.convert_bc(self.load(subpath_orig + "boundary_condition"))
        self.save(bc, subpath_new + "boundary_condition")

        # singular values
        SVs = self.load(subpath_orig + "singular_values")
        # new tenpy: bond b left  of site b, always L+1 entries
        # old tenpy: bond b right of site b, length L+1 for segment bc, others L
        SV0, SVs = SVs[0], SVs[1:]  # now entry b is right of site b
        if bc == "segment":
            SVs.append(SV0)
        # else: already correct L entries
        self.save(SVs, subpath_new + "singular_values")

        # canonical form
        form = self.load(subpath_orig + "canonical_form")
        form = [f.copy() for f in form]  # old tenpy expects list of shape (2,) arrays
        self.save(form, subpath_new + "canonical_form")

        # copy metadata
        h5gr_new.attrs["L"] = self.get_attr(h5gr_orig, "L")
        h5gr_new.attrs["max_bond_dimension"] = self.get_attr(h5gr_orig, "max_bond_dimension")

        h5gr_new.attrs["grouped"] = grouped = self.get_attr(h5gr_orig, "grouped")
        if grouped > 1:
            warnings.warn("MPS with grouped sites: "
                          "splitting afterwards not supported")
        if "site_pipes" in h5gr_orig:  # (only if it was originially converted from old tenpy)
            h5gr_new["site_pipes"] = h5gr_orig["site_pipes"]
        else:
            self.save([], subpath_new + "site_pipes")
        h5gr_new.attrs["transfermatrix_keep"] = self.get_attr(h5gr_orig, "transfermatrix_keep")
        self.save(None, subpath_new + "translate_Q1_data")

    mappings[('tenpy.networks.mps', 'MPS')] = \
        ('mps.mps', 'iMPS', convert_MPS)

    def convert_MPO(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = self.get_attr(h5gr_orig, "L")
        # tensors
        tensors = self.load(subpath_orig + "tensors")
        # convert leg labels and order
        tensors = [B.itranspose(['wL', 'wR', 'p', 'p*']).iset_leg_labels(['w', 'w*', 'p', 'p*'])
                   for B in tensors]
        self.save(tensors, subpath_new + "tensors")
        self.convert_group(h5gr_new["tensors"])  # implicitly calls self.convert_array

        # boundary conditions
        bc = self.convert_bc(self.load(subpath_orig + "boundary_condition"))
        self.save(bc, subpath_new + "boundary_condition")

        # IdL and IdR
        IdL = self.load(subpath_orig + "index_identity_left")
        IdR = self.load(subpath_orig + "index_identity_right")
        # new tenpy: bond b left  of site b, always L+1 entries
        # old tenpy: bond b right of site b, length L+1 for segment bc, others L
        IdL0, IdL = IdL[0], IdL[1:]  # now entry b is left of site b
        IdR0, IdR = IdR[0], IdR[1:]  # now entry b is left of site b
        if bc == "segment":
            IdL.append(IdL0)
            IdR.append(IdR0)
        # else: already correct L entries
        self.save(IdL, subpath_new + "index_identity_left")
        self.save(IdR, subpath_new + "index_identity_right")

        # copy metadata
        h5gr_new.attrs["L"] = self.get_attr(h5gr_orig, "L")
        h5gr_new.attrs["max_bond_dimension"] = self.get_attr(h5gr_orig, "max_bond_dimension")
        h5gr_new.attrs["grouped"] = self.get_attr(h5gr_orig, "grouped")
        self.save(None, subpath_new + "max_range")  # unknown
        self.save(None, subpath_new + "translate_Q1_data")

    mappings[('tenpy.networks.mpo', 'MPO')] = \
        ('mps.mpo', 'MPO', convert_MPO)


if __name__ == "__main__":
    args = convert.parse_args(converter_cls=Converter)
    convert.main(args)
