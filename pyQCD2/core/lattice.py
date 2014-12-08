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


def generate_halo_sites(mpi_coord, local_shape, lattice_shape, halos,
                        max_mpi_hop):
    """Generate a list of sites in the halo of the specified MPI node"""
    max_mpi_hop = (max_mpi_hop + lattice_shape.size) % lattice_shape.size
    max_mpi_hop = lattice_shape.size if max_mpi_hop == 0 else max_mpi_hop
    corner = (mpi_coord * local_shape - halos)
    halo_shape = local_shape + 2 * halos
    loc_and_halo_sites = np.array(list(np.ndindex(tuple(halo_shape))))
    # First discard any axes that don't have halos
    relevant_coords = loc_and_halo_sites[:, halos > 0]
    # Now we want to filter out all sites where there is one and only
    # one axis coordinate in the halo.
    nonzero_halos = halos[halos > 0]
    nonzero_shape = halo_shape[halos > 0]
    # This will filter when one of the coordinates is in the lower halo
    cut_behind = nonzero_halos[None, :]
    num_in_behind = (relevant_coords < cut_behind).astype(int).sum(axis=1)
    # This will filter when one of the coordinates is in the upper halo
    cut_ahead = nonzero_shape[None, :] - cut_behind - 1
    num_in_ahead = (relevant_coords > cut_ahead).astype(int).sum(axis=1)
    combined_filt = np.logical_and((num_in_ahead + num_in_behind) > 0,
                                   (num_in_ahead + num_in_behind)
                                   <= max_mpi_hop)
    return ((loc_and_halo_sites[combined_filt] + corner[None, :])
            % lattice_shape[None, :])


def compute_halo_coords(site, mpicoord, mpishape, locshape, haloshape, halos):
    """Calculates the coordinates of the given halo site in the local data"""
    mpicoord_neighb = site // locshape
    axis = mpicoord_neighb - mpicoord
    # Account for period BCs in the MPI grid
    filt = np.logical_and(axis != 0, np.abs(axis) >= mpishape / 2)
    axis[filt] = axis[filt] % (-np.sign(axis[filt]) * mpishape[filt] / 2)
    # Compute the data coordinates for the site in the node where it's not
    # a halo site
    local_coords = site % locshape + halos
    # Shift the coordinate to account for the fact it's in a halo
    local_coords[axis > 0] -= (2 * halos[axis > 0])
    local_coords[axis < 0] += halos[axis < 0]
    return local_coords % haloshape


def compute_neighbours(mpicoord, mpishape, locshape, halos, max_mpi_hop):
    """Compute the mpi coordinates of the nodes we have a halo for, then
    compute the shape of the halo for that node"""
    # N.B. Variable name prefix fnt implies that first non-zero axis offset
    # for current node is positive, whilst bck implies first non-zero axis
    # offset is negative. Each pair of variables with these prefixes should
    # be ordered so each index corresponds to nodes that are on the same
    # diagonal that passes through this node.
    ndims = halos.size
    max_mpi_hop = (ndims + max_mpi_hop) % ndims
    max_mpi_hop = ndims if max_mpi_hop == 0 else max_mpi_hop
    cart_offsets = (np.array(list(np.ndindex(tuple([3] * ndims))))
                    - np.ones(ndims, dtype=int)[None, :])
    offset_hops = np.abs(cart_offsets).sum(axis=1)
    hop_filt = np.logical_and(offset_hops <= max_mpi_hop, offset_hops > 0)
    # Filter out offsets where the mpi dimension in which the hop takes
    # place is 1, because no comm is needed in this case.
    dimension_filt = ((halos[None, :] > 0).astype(int)
                      >= np.abs(cart_offsets)).all(axis=1)
    combined_filt = np.logical_and(dimension_filt, hop_filt)
    cart_offsets = cart_offsets[combined_filt]
    fnt_cart_offsets = cart_offsets[cart_offsets.shape[0]/2:]
    # Flip the back coordinates so we align diagonals with fnt_cart_offsets
    bck_cart_offsets = cart_offsets[:cart_offsets.shape[0]/2][::-1]
    fnt_neighb_coords = ((fnt_cart_offsets + mpicoord[None, :])
                         % mpishape[None, :])
    bck_neighb_coords = ((bck_cart_offsets + mpicoord[None, :])
                         % mpishape[None, :])
    halo_buffer_shapes = np.empty(fnt_cart_offsets.shape, dtype=int)
    halo_buffer_shapes[...] = locshape
    # Filter out references to this node
    halos_broadcast = np.empty(halo_buffer_shapes.shape, dtype=int)
    halos_broadcast[...] = halos
    halo_filt = np.abs(fnt_cart_offsets) > 0
    halo_buffer_shapes[halo_filt] = halos_broadcast[halo_filt]
    return (fnt_neighb_coords, bck_neighb_coords,
            fnt_cart_offsets, bck_cart_offsets, halo_buffer_shapes)


def coord_to_index(coord, shape):
    """Converts the supplied site coordinate to a lexicographic index"""
    return reduce(lambda x, y: x * y[0] + y[1], zip(shape, coord), 0)


class Lattice(object):
    """Handles MPI allocation of lattice sites"""

    def __init__(self, shape, halo=1, max_mpi_hop=-1):
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
        self.local_site_coords = generate_local_sites(self.mpicoord,
                                                      self.locshape)
        self.local_site_indices = np.array([coord_to_index(x, self.latshape)
                                            for x in self.local_site_coords])
        self.halo_site_coords = generate_halo_sites(self.mpicoord,
                                                    self.locshape,
                                                    self.latshape, self.halos,
                                                    max_mpi_hop)
        self.halo_site_indices = np.array([coord_to_index(x, self.latshape)
                                           for x in self.halo_site_coords])
        # Compute neighbour coordinates
        neighbour_info = compute_neighbours(self.mpicoord, self.mpishape,
                                            self.locshape, self.halos,
                                            max_mpi_hop)
        self.fnt_neighb_coords = neighbour_info[0]
        self.bck_neighb_coords = neighbour_info[1]
        self.fnt_halo_norms = neighbour_info[2]
        self.bck_halo_norms = neighbour_info[3]
        self.halo_buffer_shapes = neighbour_info[4]
        self.fnt_neighb_ranks = np.array(map(self.comm.Get_cart_rank,
                                             self.fnt_neighb_coords))
        self.bck_neighb_ranks = np.array(map(self.comm.Get_cart_rank,
                                             self.bck_neighb_coords))

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

    def message(self, msg):
        if self.comm.Get_rank() == 0:
            print(msg)

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
        slices = [slice(h, -h) if (h > 0 and i > dim) else slice(None)
                  for i, h in enumerate(self.halos)]
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

    def make_halo_buffers(self, data):
        """Make buffers for the halo swap function"""
        send_buffers = []
        recv_buffers = []
        for i in range(self.ndims):
            if self.halos[i] == 0:
                # Don't need a buffer as there's no halo
                send_buffers.append([])
                recv_buffers.append([])
                continue
            shape = np.zeros(self.ndims, int)
            shape[:i] = self.haloshape[:i]
            shape[i:] = self.locshape[i:]
            shape[i] = self.halos[i]
            snd_bufs_i = []
            rcv_bufs_i = []
            for d in [1, -1]:
                slicer = self.halo_slice(i, d, 'send')
                snd_bufs_i.append(data[slicer].copy())
                rcv_bufs_i.append(np.empty(tuple(shape), dtype=data.dtype))
            send_buffers.append(snd_bufs_i)
            recv_buffers.append(rcv_bufs_i)
        return send_buffers, recv_buffers

    def buffers_to_data(self, data, recv_buffers):
        """Puts the received buffer data in the data array"""
        for i in range(self.ndims):
            for d, buf in zip([1, -1], recv_buffers[i]):
                slicer = self.halo_slice(i, d, 'recv')
                data[slicer] = buf

    def get_site_rank(self, site):
        """Gets the rank of the node in which the specified site lies"""
        mpicoords = site // self.locshape
        return self.comm.Get_cart_rank(mpicoords)

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