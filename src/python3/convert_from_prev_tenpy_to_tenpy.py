#!/usr/bin/env python3
"""Convert the format of hdf5 files from the TeNPy precursor to the new TeNPy."""
# Try command line argument ``--help`` for options.

import convert
from convert import Hdf5Converter
import hdf5_io

import tenpy  # requirement: can import the new tenpy
import tenpy.linalg.np_conserved as npc
import numpy as np


class Converter(Hdf5Converter):
    """Convert from the `prev_tenpy` format to the new `tenpy`."""

    from_format = "prev_tenpy"  #: from which fromat the converter converts
    to_format = "tenpy"  #: into which format the converter converts

    mappings = {}
    mappings[('tools.hdf5_io', 'Hdf5Exportable')] = \
        ('tenpy.tools.hdf5_io', 'Hdf5Exportable', Hdf5Converter.convert_Hdf5Exportable)

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

    mappings[('algorithms.linalg.np_conserved', 'array')] = \
        ('tenpy.linalg.np_conserved', 'Array', convert_array)

    def convert_bc(self, bc):
        """MPS/MPO boundary conditions."""
        # prev_tenpy: bc = 'finite', 'segment', 'periodic'
        # new  tenpy:      'finite', 'segment', 'infinite'
        if bc == 'periodic':
            bc = 'infinite'
        if bc not in ['finite', 'segment', 'infinite']:
            raise ValueError("unexpected boundary conditions bc={0!r}".format(bc))
        return bc

    def convert_MPS(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = h5gr_orig.attrs["L"]
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
        # old tenpy: singular values indexed by bond b right of site b
        # entry 0 always right of site 0, length of SVs depends on bc
        # need new tenpy: always L+1 entries, 0 left of site 0
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
        h5gr_new.attrs["L"] = h5gr_orig.attrs["L"]
        h5gr_new.attrs["max_bond_dimension"] = h5gr_orig.attrs["max_bond_dimension"]

        h5gr_new.attrs["norm"] = 1 # required by new TeNPy
        h5gr_new.attrs["grouped"] = grouped = h5gr_orig.attrs["grouped"]
        if grouped > 1:
            warnings.warn("MPS with grouped sites: "
                          "splitting afterwards is not supported in new TeNPy")
            # still copy site_pipes: allow converting back to prev_tenpy for splitting
            h5gr_new["site_pipes"] = h5gr_old["site_pipes"]
        h5gr_new.attrs["_transfermatrix_keep"] = h5gr_orig.attrs["transfermatrix_keep"]


    mappings[('mps.mps', 'iMPS')] = \
        ('tenpy.networks.mps', 'MPS', convert_MPS)

    def convert_MPO(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = h5gr_orig.attrs["L"]
        if h5gr_orig["translate_Q1_data"].attrs[hdf5_io.ATTR_TYPE] != hdf5_io.REPR_NONE:
            raise ValueError("Can't convert MPO with non-trivial 'translate_Q1_data'")
        # tensors: (implicitly) call above conversion function
        self.convert_group(h5gr_orig["tensors"])  # convert the arrays
        tensors = self.load(subpath_orig + "tensors")
        # convert leg labels. In prev_tenpy, the order is guaranteed.
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
        # old tenpy: indexed by bond b right of site b
        # entry 0 always right of site 0, length of IdL depends on bc
        # new tenpy: always L+1 entries, 0 left of site 0
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

        # copy metadata
        h5gr_new.attrs["L"] = h5gr_orig.attrs["L"]
        h5gr_new.attrs["max_bond_dimension"] = h5gr_orig.attrs["max_bond_dimension"]
        h5gr_new.attrs["grouped"] = grouped = h5gr_orig.attrs["grouped"]
        self.save(None, subpath_new + "max_range")  # unknown

    mappings[('mps.mpo', 'MPO')] = \
        ('tenpy.networks.mpo', 'MPO', convert_MPO)


if __name__ == "__main__":
    args = convert.parse_args(converter_cls=Converter)
    convert.main(args)
