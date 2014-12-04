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
        data_shape = (tuple(map(lambda x: x + 2 * lattice.halo,
                                lattice.locshape))
                      + field_shape)
        self.data = np.zeros(data_shape, dtype=dtype)

    def halo_swap(self):
        pass

    def roll(self, axis, nsites):
        pass

    def __getitem__(self, *args):
        pass

    def __mul__(self, other):
        pass