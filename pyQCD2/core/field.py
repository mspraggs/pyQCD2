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

    def halo_swap(self, axis=None):
        """Swap the values in the field values in the halos between adjacent
        nodes using MPI"""

        comm = self.lattice.comm
        halos = self.lattice.halos

        # If there's only one node, don't bother swapping
        if comm.Get_size() == 1:
            return
        for i, neighbours in enumerate(self.lattice.mpi_neighbours):
            if not neighbours:
                continue
            back, front = neighbours
            send_slice = [slice(halo, -halo) if halo > 1 else slice(None)
                          for halo in halos]
            recv_slice = [slice(halo, -halo) if halo > 1 else slice(None)
                          for halo in halos]
            # First pass data forwards
            send_slice[i] = slice(-2 * halos[i], -halos[i])
            recv_slice[i] = slice(None, halos[i])
            buffer = self.data[tuple(send_slice)].copy()
            print(self.data)
            comm.Send([buffer, self.mpi_dtype], dest=front)
            comm.Recv([buffer, self.mpi_dtype], source=back)
            self.data[tuple(recv_slice)] = buffer
            print(self.data)
            # Now pass data backwards
            send_slice[i] = slice(halos[i], 2 * halos[i])
            recv_slice[i] = slice(-halos[i], None)
            buffer = self.data[tuple(send_slice)].copy()
            comm.Send([buffer, self.mpi_dtype], dest=back)
            comm.Recv([buffer, self.mpi_dtype], source=front)
            self.data[tuple(recv_slice)] = buffer

    def roll(self, axis, nsites):
        pass

    def __getitem__(self, *args):
        pass

    def __mul__(self, other):
        pass