"""Test Field class"""

from __future__ import absolute_import

import collections

from mpi4py import MPI
import numpy as np
import pytest

from pyQCD2.core.lattice import Lattice
from pyQCD2.core.field import Field


@pytest.fixture
def test_field():
    """Fixture to set up a scalar field object"""
    FieldFixture = collections.namedtuple("FieldFixture",
                                          ["field", "params"])
    lattice = Lattice((8, 4, 4, 4))
    field = Field(lattice, (), float)
    start = lattice.locvol * lattice.comm.Get_rank()
    end = lattice.locvol * (lattice.comm.Get_rank() + 1)
    s = slice(lattice.halo, -lattice.halo)
    field.data[s, s, s, s] = 1.0
    return FieldFixture(field,
                        dict(field_shape=(), dtype=float,
                             mpi_dtype=MPI.DOUBLE,
                             lattice=lattice))


class TestField(object):

    def test_init(self, test_field):
        """Test Field constructor"""
        field = test_field.field
        for key, value in test_field.params.items():
            assert getattr(field, key) == value
        lattice = test_field.params['lattice']
        expected_shape = (tuple([N + 2 * lattice.halo
                                 for N in lattice.locshape])
                          + test_field.params['field_shape'])
        assert field.data.shape == expected_shape

    def test_halo_swap(self, test_field):
        """Test for halo swapping"""

        # Exit if there's no MPI involved
        if test_field.params['lattice'].comm.Get_size() == 1:
            return

        # First check that all data is zero
        ndims = test_field.params['lattice'].ndims
        for dim in range(ndims):
            for i in [0, -1]:
                selector = [slice(None)] * ndims
                selector[dim] = i
                selector = tuple(selector)
                assert np.allclose(test_field.field.data[selector], 0)
        test_field.field.halo_swap()
        for dim in range(ndims):
            for i in [0, -1]:
                selector = [slice(None)] * ndims
                selector[dim] = i
                selector = tuple(selector)
                assert np.allclose(test_field.field.data[selector], 1)