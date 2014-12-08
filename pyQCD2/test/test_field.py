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
    s = tuple([slice(h, -h) if h > 0 else slice(None) for h in lattice.halos])
    field.data[s] = 1.0
    return FieldFixture(field,
                        dict(field_shape=(), dtype=float,
                             mpi_dtype=MPI.DOUBLE,
                             lattice=lattice))


@pytest.fixture
def full_halo_field():
    """Fixture to set up a lattice and field with a full halo"""
    FieldFixture = collections.namedtuple("FieldFixture",
                                          ["field", "lattice"])
    lattice = Lattice((8, 4, 4, 4), max_mpi_hop=4)
    field = Field(lattice, (), float)
    s = tuple([slice(h, -h) if h > 0 else slice(None) for h in lattice.halos])
    field.data[s] = 1.0
    return FieldFixture(field, lattice)


class TestField(object):

    def test_init(self, test_field):
        """Test Field constructor"""
        field = test_field.field
        for key, value in test_field.params.items():
            assert getattr(field, key) == value
        lattice = test_field.params['lattice']
        haloshape = tuple([x + 2 * y
                           for x, y in zip(lattice.locshape,
                                           lattice.halos)])
        expected_shape = (haloshape + test_field.params['field_shape'])
        assert field.data.shape == expected_shape

    def test_halo_swap(self, full_halo_field):
        """Test for halo swapping"""

        # Exit if there's no MPI involved
        if full_halo_field.lattice.comm.Get_size() == 1:
            return

        # First check that all data is zero
        ndims = full_halo_field.lattice.ndims
        halos = full_halo_field.lattice.halos
        for dim in range(ndims):
            if halos[dim] == 0:
                continue
            selector = [slice(halo, -halo) if halo > 0 else slice(None)
                        for halo in halos]
            selector[dim] = slice(None, halos[dim])
            assert np.allclose(full_halo_field.field.data[tuple(selector)], 0)
            selector[dim] = slice(-halos[dim], None)
            assert np.allclose(full_halo_field.field.data[tuple(selector)], 0)
        full_halo_field.field.halo_swap()
        
        assert np.allclose(full_halo_field.field.data, 1)

    def test_halo_swap_nonblock(self, full_halo_field):

        # Exit if there's no MPI involved
        if full_halo_field.lattice.comm.Get_size() == 1:
            return

        # First check that all data is zero
        ndims = full_halo_field.lattice.ndims
        halos = full_halo_field.lattice.halos
        for dim in range(ndims):
            if halos[dim] == 0:
                continue
            selector = [slice(halo, -halo) if halo > 0 else slice(None)
                        for halo in halos]
            selector[dim] = slice(None, halos[dim])
            assert np.allclose(full_halo_field.field.data[tuple(selector)], 0)
            selector[dim] = slice(-halos[dim], None)
            assert np.allclose(full_halo_field.field.data[tuple(selector)], 0)
        recv_buffers = full_halo_field.field.halo_swap(block=False)
        full_halo_field.lattice.comm.Barrier()
        full_halo_field.lattice.buffers_to_data(full_halo_field.field.data,
                                                recv_buffers)
        assert np.allclose(full_halo_field.field.data, 1)

    def test_fill(self, test_field):
        """Test Field.fill"""

        test_field.field.fill(2.0)
        assert np.allclose(test_field.field.data, 2.0)

        with pytest.raises(ValueError):
            test_field.field.fill(np.arange(5))