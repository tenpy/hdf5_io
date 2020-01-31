#!/usr/bin/env python3
# Try command line argument ``--help`` for options.

import convert
from convert import Hdf5Converter
import hdf5_io

import tenpy  # requirement: can import the new tenpy
import tenpy.linalg.np_conserved as npc
import numpy as np


class Converter(Hdf5Converter):
    """Convert from the "old" TeNpy (in python 2) to the "new" TeNPy."""

    mappings = {}
    mappings[('tools.hdf5_io', 'Hdf5Exportable')] = \
        ('tenpy.tools.hdf5_io', 'Hdf5Exportable', Hdf5Converter.convert_Hdf5Exportable)

    def convert_array(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        # copy (hardlinks avoid copies)
        h5gr_new["dtype"] = h5gr_orig["dtype"]
        h5gr_new["blocks"] = h5gr_orig["blocks"]
        total_charge = self.get_attr(h5gr_orig, "total_charge")
        self.save(total_charge, subpath_new + "total_charge")
        block_inds = np.asarray(self.load(subpath_orig + "block_inds"), np.intp)
        self.save(block_inds, subpath_new + "block_inds")
        h5gr_new.attrs["block_inds_sorted"] = h5gr_orig.attrs["block_inds_sorted"]
        h5gr_new.attrs["rank"] = rank = self.get_attr(h5gr_orig, "rank")
        h5gr_new.attrs["shape"] = self.get_attr(h5gr_orig, "shape")

        # convert charges
        num_q = self.get_attr(h5gr_orig, "num_charges")
        mod_q = self.get_attr(h5gr_orig, "U1_ZN")
        leg_charges = self.load(subpath_orig + "leg_charges")
        qconj = self.get_attr(h5gr_orig, "qconj")

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
        self.save(chinfo, subpath_new + "chinfo")
        self.save(legs_new, subpath_new + "legs")

        labels = self.load(subpath_orig + "labels")
        # convert {str: int} -> [str]
        labels_new = [None] * rank
        if labels is not None:
            for k, v in labels.items():
                labels_new[v] = k
        self.save(labels_new, subpath_new + "labels")

    mappings[('algorithms.linalg.np_conserved', 'array')] = \
        ('tenpy.linalg.np_conserved', 'Array', convert_array)

    def convert_bc(self, bc):
        """MPS/MPO boundary conditions."""
        # old tenpy: bc = 'finite', 'segment', 'periodic'
        # new tenpy:      'finite', 'segment', 'infinite'
        if bc == 'periodic':
            bc = 'infinite'
        if bc not in ['finite', 'segment', 'infinite']:
            raise ValueError("unexpected boundary conditions bc={0!r}".format(bc))
        return bc

    def convert_MPS(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = self.get_attr(h5gr_orig, "L")
        if h5gr_orig["translate_Q1_data"].attrs[hdf5_io.ATTR_TYPE] != hdf5_io.REPR_NONE:
            raise ValueError("Can't convert MPS with non-trivial 'translate_Q1_data'")

        # tensors
        self.convert_group(h5gr_orig["tensors"])  # implicitly calls self.convert_array
        tensors = self.load(subpath_orig + "tensors")
        # convert leg labels and order
        tensors = [B.iset_leg_labels(['p', 'vL', 'vR']).itranspose(['vL', 'p', 'vR'])
                   for B in tensors]
        chinfo = tensors[0].legs[0].chinfo
        for B in tensors:
            for leg in B.legs:
                leg.chinfo = chinfo  # share chinfo
        self.save(tensors, subpath_new + "tensors")
        self.save(chinfo, subpath_new + "chinfo")
        # make trivial sites with just the physical leg
        sites = [tenpy.networks.site.Site(B.legs[1]) for B in tensors]
        self.save(sites, subpath_new + "sites")

        # boundary conditions
        bc = self.convert_bc(self.load(subpath_orig + "boundary_condition"))
        self.save(bc, subpath_new + "boundary_condition")

        # singular values
        SVs = self.load(subpath_orig + "singular_values")
        # new tenpy: bond b left  of site b, always L+1 entries
        # old tenpy: bond b right of site b, length L+1 for segment bc, others L
        SVs = [None] + SVs  # now entry b is left of site b
        # fine for infinite bc with L+1 entries
        if bc == 'infinite':
            SVs[0] = SVs[-1]
        elif bc == 'finite':
            SVs[0] = SVs[-1] = np.ones([1])
        else: # segment
            SVs[0] = SVs.pop()
        self.save(SVs, subpath_new + "singular_values")

        # canonical form
        form = self.load(subpath_orig + "canonical_form")
        if type(form) == np.ndarray:
            assert form.shape == (2,)
            form = np.array([form]*L)  # old tenpy with single 2-tuple, valid for all sites
        else:  # old tenpy: list of array
            form = np.array(form)
            assert form.shape == (L, 2)
        self.save(form, subpath_new + "canonical_form")

        # copy metadata
        h5gr_new.attrs["L"] = self.get_attr(h5gr_orig, "L")
        h5gr_new.attrs["max_bond_dimension"] = self.get_attr(h5gr_orig, "max_bond_dimension")

        h5gr_new.attrs["norm"] = 1 # required by new TeNPy
        h5gr_new.attrs["grouped"] = grouped = self.get_attr(h5gr_orig, "grouped")
        if grouped > 1:
            warnings.warn("MPS with grouped sites: "
                          "splitting after conversion not supported")
        # still copy site_pipes: allow converting back to old tenpy for splitting
        h5gr_new["site_pipes"] = h5gr_orig["site_pipes"]
        h5gr_new.attrs["transfermatrix_keep"] = self.get_attr(h5gr_orig, "transfermatrix_keep")


    mappings[('mps.mps', 'iMPS')] = \
        ('tenpy.networks.mps', 'MPS', convert_MPS)

    def convert_MPO(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = self.get_attr(h5gr_orig, "L")
        if self.get_attr(h5gr_orig["translate_Q1_data"], hdf5_io.ATTR_TYPE) != hdf5_io.REPR_NONE:
            raise ValueError("Can't convert MPO with non-trivial 'translate_Q1_data'")
        # tensors: (implicitly) call above conversion function
        self.convert_group(h5gr_orig["tensors"])  # convert the arrays
        tensors = self.load(subpath_orig + "tensors")
        # convert leg labels. In old tenpy, the order is guaranteed.
        tensors = [W.iset_leg_labels(['wL', 'wR', 'p', 'p*']) for W in tensors]
        chinfo = tensors[0].legs[0].chinfo
        for W in tensors:
            for leg in W.legs:
                leg.chinfo = chinfo  # share chinfo
        p_legs = [W.legs[2] for W in tensors]
        self.save(tensors, subpath_new + "tensors")
        self.save(chinfo, subpath_new + "chinfo")
        # make trivial sites with just the physical leg
        sites = [tenpy.networks.site.Site(W.legs[2]) for W in tensors]
        self.save(sites, subpath_new + "sites")

        # boundary conditions
        bc = self.convert_bc(self.load(subpath_orig + "boundary_condition"))
        self.save(bc, subpath_new + "boundary_condition")

        # IdL and IdR
        IdL = list(self.load(subpath_orig + "index_identity_left"))
        IdR = list(self.load(subpath_orig + "index_identity_right"))
        # new tenpy: bond b left  of site b, always L+1 entries
        # old tenpy: bond b right of site b, length L+1 for segment bc, others L
        IdL = [None] + IdL  # now entry b is left of site b
        IdR = [None] + IdR
        if bc == 'infinite':
            IdL[0] = IdL[-1]
            IdR[0] = IdR[-1]
        elif bc == 'finite':
            IdL[0] = IdR[-1] = 0
            IdL[0] = IdR[-1] = 0
        else: # segment
            IdL[0] = IdL.pop()
            IdR[0] = IdR.pop()
        self.save(IdL, subpath_new + "index_identity_left")
        self.save(IdR, subpath_new + "index_identity_right")

        # copy metadata
        h5gr_new.attrs["L"] = self.get_attr(h5gr_orig, "L")
        h5gr_new.attrs["max_bond_dimension"] = self.get_attr(h5gr_orig, "max_bond_dimension")
        h5gr_new.attrs["grouped"] = self.get_attr(h5gr_orig, "grouped")
        self.save(None, subpath_new + "max_range")  # unknown

    mappings[('mps.mpo', 'MPO')] = \
        ('tenpy.networks.mpo', 'MPO', convert_MPO)


if __name__ == "__main__":
    args = convert.parse_args(converter_cls=Converter)
    convert.main(args)
