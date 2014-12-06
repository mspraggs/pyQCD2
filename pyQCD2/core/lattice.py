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
    corner = mpi_coord * local_shape
    local_sites = np.array(list(np.ndindex(tuple(local_shape))))
    return local_sites + corner[None, :]
    return [tuple(site) for site in (local_sites + corner[None, :])]


def generate_halo_sites(mpi_coord, local_shape, lattice_shape, halos):
    """Generate a list of sites in the halo of the specified MPI node"""
    corner = (mpi_coord * local_shape - halos)
    halo_shape = local_shape + 2 * halos
    loc_and_halo_sites = np.array(list(np.ndindex(tuple(halo_shape))))
    # First discard any axes that don't have halos
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
    return ((loc_and_halo_sites[combined_filt] + corner[None, :])
            % lattice_shape[None, :])


def compute_halo_coords(site, mpicoord, mpishape, locshape, haloshape, halos):
    """Calculates the coordinates of the given halo site in the local data"""
    mpicoord_neighb = site // locshape
    axis = mpicoord_neighb - mpicoord
    filt = axis != 0
    axisf = axis[filt]
    # Account for period BCs in the MPI grid
    axis[filt] = (axisf if (np.abs(axisf) < mpishape[filt] / 2)
                  else axisf % (-np.sign(axisf) * mpishape[filt] / 2))
    # Compute the data coordinates for the site in the node where it's not
    # a halo site
    local_coords = site % locshape + halos
    # Shift the coordinate to account for the fact it's in a halo
    halo_shift = (-2 * halos) if (axis > 0).any() else halos
    local_coords += halo_shift * filt
    return local_coords % haloshape


class Lattice(object):
    """Handles MPI allocation of lattice sites"""

    def __init__(self, shape, halo=1):
        """Construct Lattice object for lattice with supplied shape"""
        nprocs = MPI.COMM_WORLD.Get_size()
        self.latshape = np.array(shape)
        self.mpishape = np.array(MPI.Compute_dims(nprocs, len(shape)))
        self.locshape = self.latshape // self.mpishape
        self.halos = halo * (self.mpishape > 1).astype(int)
        self.haloshape = self.locshape + 2 * self.halos
        self.locvol = self.locshape.prod(0)
        self.latvol = self.latshape.prod(0)
        self.ndims = self.latshape.size
        self.nprocs = nprocs

        self.comm = MPI.COMM_WORLD.Create_cart(self.mpishape)
        if (self.latshape % self.mpishape).sum() > 0:
            raise RuntimeError("Invalid number of MPI processes")

        # Determine the coordinates of the sites on the current node
        self.mpicoord = np.array(self.comm.Get_coords(self.comm.Get_rank()))
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
        site = np.array(site)[None, :]
        return ((self.local_sites == site).all(axis=1).any()
                or (self.halo_sites == site).all(axis=1).any())

    def halo_slice(self, dim, position, send_recv):
        """Generate the slice specifiying the halo region of Field.data"""
        # dim - dimension/axis of the slice
        # location - whether front (+ve direction, +1) or rear (-ve direction,
        # -1) halo is being used.
        # send_recv - whether the slice is the halo itself ('recv') or the
        # selection of local sites corresponding to a halo on another node
        # ('send')
        slices = [slice(h, -h) if h > 0 else slice(None)
                  for h in self.halos]
        if position < 0 and send_recv == 'send':
            slices[dim] = slice(self.halos[dim], 2 * self.halos[dim])
        elif position < 0 and send_recv == 'recv':
            slices[dim] = slice(None, self.halos[dim])
        elif position > 0 and send_recv == 'send':
            slices[dim] = slice(-2 * self.halos[dim], -self.halos[dim])
        elif position > 0 and send_recv == 'recv':
            slices[dim] = slice(-self.halos[dim], None)
        else:
            pass
        return tuple(slices)

    def get_local_coords(self, site):
        """Get the local coordinates of the specified site"""
        if (self.local_sites == site[None, :]).all(axis=1).any():
            # Account for the halo around the local data
            local_coords = site % self.locshape
            return tuple(local_coords + self.halos)
        elif (self.halo_sites == site[None, :]).all(axis=1).any():
            return compute_halo_coords(site, self.mpicoord, self.mpishape,
                                       self.locshape, self.haloshape,
                                       self.halos)
        else:
            return None

    def get_local_index(self, site):
        """Get the local lexicographic index of the specified site"""
        local_coords = self.get_local_coords(site)
        if local_coords is not None:
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