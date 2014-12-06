"""
This module contains the Lattice class, which handles the MPI layout of the
lattice.
"""

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI
import numpy as np


def generate_local_sites(mpi_coord, local_shape):
    """Generate list of sites local to the specified MPI node"""
    corner = np.array(mpi_coord) * np.array(local_shape)
    local_sites = np.array(list(np.ndindex(local_shape)))
    return [tuple(site) for site in (local_sites + corner[None, :])]


def generate_halo_sites(mpi_coord, local_shape, lattice_shape, halos):
    """Generate a list of sites in the halo of the specified MPI node"""
    lattice_shape = np.array(lattice_shape)
    corner = (np.array(mpi_coord) * np.array(local_shape) - np.array(halos))
    halo_shape = np.array(local_shape) + 2 * np.array(halos)
    loc_and_halo_sites = np.array(list(np.ndindex(tuple(halo_shape))))
    # First discard any axes that don't have halos
    halos = np.array(halos)
    relevant_coords = loc_and_halo_sites[:, halos > 0]
    # Now we want to filter out all sites where there isn't one and only
    # one axis coordinate in the halo.
    nonzero_halos = halos[halos > 0]
    nonzero_shape = halo_shape[halos > 0]
    # This will filter when one of the coordinates is in the lower halo
    cut_behind = nonzero_halos[None, :]
    num_in_behind = (relevant_coords < cut_behind).astype(int).sum(axis=1)
    # This will filter when one of the coordinates is in the upper halo
    cut_ahead = nonzero_shape[None, :] - cut_behind - 1
    num_in_ahead = (relevant_coords > cut_ahead).astype(int).sum(axis=1)
    combined_filt = (num_in_ahead + num_in_behind) == 1
    return [tuple((site + corner) % lattice_shape)
            for site in loc_and_halo_sites[combined_filt]]


class Lattice(object):
    """Handles MPI allocation of lattice sites"""

    def __init__(self, shape, halo=1):
        """Construct Lattice object for lattice with supplied shape"""
        nprocs = MPI.COMM_WORLD.Get_size()
        self.mpishape = tuple(MPI.Compute_dims(nprocs, len(shape)))
        self.locshape = tuple([x // y for x, y in zip(shape, self.mpishape)])
        self.halos = map(lambda x: (x > 1) * halo, self.mpishape)
        self.haloshape = tuple(map(lambda x: x[0] + 2 * x[1],
                                   zip(self.locshape, self.halos)))
        self.latshape = shape
        self.locvol = reduce(lambda x, y: x * y, self.locshape)
        self.latvol = reduce(lambda x, y: x * y, self.latshape)
        self.ndims = len(shape)
        self.nprocs = nprocs

        self.comm = MPI.COMM_WORLD.Create_cart(self.mpishape)
        remainders = [x % y for x, y in zip(shape, self.mpishape)]
        if sum(remainders) > 0:
            raise RuntimeError("Invalid number of MPI processes")

        # Determine the coordinates of the sites on the current node
        self.mpicoord = tuple(self.comm.Get_coords(self.comm.Get_rank()))
        self.local_sites = generate_local_sites(self.mpicoord, self.locshape)
        self.halo_sites = generate_halo_sites(self.mpicoord, self.locshape,
                                              self.latshape, self.halos)

        self.mpi_neighbours = []
        for dim in range(self.ndims):
            axis_neighbours = []
            for offset in [-1, 1]:
                coord = list(self.mpicoord)
                coord[dim] = (coord[dim] + offset) % self.mpishape[dim]
                neighbour_rank = self.comm.Get_cart_rank(coord)
                if neighbour_rank == self.comm.Get_rank():
                    axis_neighbours = []
                    break
                axis_neighbours.append(neighbour_rank)
            self.mpi_neighbours.append(axis_neighbours)

    def ishere(self, site):
        """Determine whether the current coordinate is here"""
        return site in self.local_sites + self.halo_sites

    def get_local_coords(self, site):
        """Get the local coordinates of the specified site"""
        if site in self.local_sites:
            # Account for the halo around the local data
            corner = np.array(self.halos)
            local_coords = np.array(self.sanitize(site, self.locshape))
            return tuple(local_coords + corner)
        elif site in self.halo_sites:
            site = np.array(site)
            mpishape = np.array(self.mpishape)
            mpicoord = site // np.array(self.locshape)
            axis = mpicoord - np.array(self.mpicoord)
            filt = axis != 0
            axisf = axis[filt]
            axisf = (axisf if (np.abs(axisf) < mpishape[filt] / 2)
                     else axisf % (np.sign(axisf) * mpishape[filt]))
            local_coords = np.array(self.sanitize(site, self.locshape))
            local_coords += ((3 if (axis > 0).any() else 1)
                             * axis * np.array(self.halos))
            return self.sanitize(tuple(local_coords), self.haloshape)
        else:
            return None

    def get_local_index(self, site):
        """Get the local lexicographic index of the specified site"""
        local_coords = self.get_local_coords(site)
        if local_coords:
            it = zip(self.haloshape[1:], local_coords[1:])
            return reduce(lambda x, y: x * y[0] + y[1], it,
                          local_coords[0])
        else:
            return None

    @staticmethod
    def sanitize(site, shape):
        """Applies periodic boundary conditions to the given site coordinate
        using the given the lattice shape"""
        return tuple(map(lambda x: x[0] % x[1], zip(site, shape)))