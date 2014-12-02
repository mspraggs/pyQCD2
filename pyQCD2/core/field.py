"""
This module contains code common to all field types
"""

from __future__ import absolute_import


class Field(object):

    def __init__(self, lattice, local_shape):
        pass

    def ishere(self, site):
        pass

    def halo_swap(self):
        pass

    def roll(self, axis, nsites):
        pass

    def __getitem__(self, *args):
        pass

    def __mul__(self, other):
        pass