"""
This module contains code common to all field types
"""

from __future__ import absolute_import


class Field(object):
    """Base Field class, from which all other fields are derived"""

    def __init__(self, lattice, field_shape, dtype):
        """Field constructor"""
        self.lattice = lattice
        self.field_shape = field_shape

        self.data = np.zeros(lattice.locshape + field_shape, dtype=dtype)

    def halo_swap(self):
        pass

    def roll(self, axis, nsites):
        pass

    def __getitem__(self, *args):
        pass

    def __mul__(self, other):
        pass