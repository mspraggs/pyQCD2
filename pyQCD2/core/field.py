"""
This module contains code common to all field types
"""

from __future__ import absolute_import

from mpi4py import MPI
import numpy as np

type_lookup = {np.dtype('float32'): MPI.FLOAT,
               np.dtype('float64'): MPI.DOUBLE,
               np.dtype('complex64'): MPI.COMPLEX8,
               np.dtype('complex128'): MPI.COMPLEX16}


class Field(object):
    """Base Field class, from which all other fields are derived"""

    def __init__(self, lattice, field_shape, dtype):
        """Field constructor"""
        self.lattice = lattice
        self.field_shape = field_shape
        self.dtype = np.dtype(dtype)
        self.mpi_dtype = type_lookup[self.dtype]
        self.data = np.zeros(lattice.haloshape, dtype=dtype)

    def fill(self, value):
        """Fill all site values with the specified value"""

    def halo_swap(self, axis=None, buffers=None, block=True):
        """Swap the values in the field values in the halos between adjacent
        nodes using MPI"""

        if buffers:
            send_buffers, recv_buffers = buffers
        else:
            send_buffers, recv_buffers = self.lattice.make_halo_buffers(self.data)

        comm = self.lattice.comm
        # If there's only one node, don't bother swapping
        if comm.Get_size() == 1:
            return

        for i, ranks in enumerate(zip(self.lattice.fnt_neighb_ranks,
                                      self.lattice.bck_neighb_ranks)):
            for j, direc in enumerate([1, -1]):
                node_to, node_from = ranks[::direc]
                comm.Isend([send_buffers[i][j], self.mpi_dtype],
                           dest=node_to)
                comm.Irecv([recv_buffers[i][j], self.mpi_dtype],
                           source=node_from)
        # If blocking, wait for processes to finish and fill the data variable
        if block:
            comm.Barrier()
            self.lattice.buffers_to_data(self.data, recv_buffers)
        else:
            return recv_buffers

    def fill(self, value):
        """Fill the field with the specified value"""

        if np.shape(value) != self.field_shape:
            raise ValueError("Supplied value does not have shape {}"
                             .format(self.field_shape))
        data_slice = [slice(None)] * self.lattice.ndims
        self.data[data_slice] = value

    def roll(self, axis, nsites):
        pass

    def __getitem__(self, *args):
        pass

    def __mul__(self, other):
        pass