"""
This module contains the Lattice class, which handles the MPI layout of the
lattice.
"""

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI
import numpy as np


class Lattice(object):
    """Handles MPI allocation of lattice sites"""

    def __init__(self, shape, halo=1):
        """Construct Lattice object for lattice with supplied shape"""
        nprocs = MPI.COMM_WORLD.Get_size()
        self.mpishape = tuple(MPI.Compute_dims(nprocs, len(shape)))
        self.locshape = tuple([x // y for x, y in zip(shape, self.mpishape)])
        self.haloshape = tuple(map(lambda x: x + 2 * halo, self.locshape))
        self.latshape = shape
        self.locvol = reduce(lambda x, y: x * y, self.locshape)
        self.latvol = reduce(lambda x, y: x * y, self.latshape)
        self.ndims = len(shape)
        self.nprocs = nprocs
        self.halo = halo

        self.comm = MPI.COMM_WORLD.Create_cart(self.mpishape)
        remainders = [x % y for x, y in zip(shape, self.mpishape)]
        if sum(remainders) > 0:
            raise RuntimeError("Invalid number of MPI processes")

        # Determine the coordinates of the sites on the current node
        self.mpicoord = tuple(self.comm.Get_coords(self.comm.Get_rank()))
        # Corner of the lattice on this node
        corner = tuple([x * y for x, y in zip(self.mpicoord, self.locshape)])
        self.local_sites = (np.array(list(np.ndindex(self.locshape)))
                            + np.array([corner])).tolist()
        self.local_sites = map(lambda site: tuple(site), self.local_sites)

    def ishere(self, site):
        """Determine whether the current coordinate is here"""
        site = tuple(map(lambda x: x[0] % x[1], zip(site, self.latshape)))
        return site in self.sites