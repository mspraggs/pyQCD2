"""
This module contains the code specific to GaugeField objects.
"""

from __future__ import absolute_import

import numpy as np

from pyQCD2.lib.field_linalg import mult_gauge_gauge
from pyQCD2.core import field


class GaugeField(field.Field):
    """Gauge field class for SU(N) gauge fields"""
    
    def __init__(self, lattice, nc, data=None):

        self.nc = nc
        self.lattice = lattice
        if data is not None:
            self.data = data.reshape(lattice.locshape
                                     + (lattice.ndims, nc, nc))
        else:
            self.data = np.zeros(lattice.locshape
                                 + (lattice.ndims, nc, nc))

    def __mul__(self, other):
        flat_shape = (self.lattice.locvol, self.lattice.ndims,
                      self.nc, self.nc)
        ours_flat = self.data.reshape(flat_shape)
        theirs_flat = other.reshape(flat_shape)
        out = mult_gauge_gauge(ours_flat, theirs_flat)
        return GaugeField(self.lattice, self.nc, out)
