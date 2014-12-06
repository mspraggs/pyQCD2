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
            for direc in [1, -1]:
                # direc = 1 -> pass forward; direc = -1 -> pass backward
                # neighbours = [node_behind, node_ahead]
                node_from, node_to = neighbours[::direc]
                send_slicer = self.lattice.halo_slice(i, direc, 'send')
                recv_slicer = self.lattice.halo_slice(i, -direc, 'recv')
                buffer = self.data[send_slicer].copy()
                comm.Send([buffer, self.mpi_dtype], dest=node_to)
                comm.Recv([buffer, self.mpi_dtype], source=node_from)
                self.data[recv_slicer] = buffer

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