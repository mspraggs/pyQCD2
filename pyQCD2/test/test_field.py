"""Test Field class"""

from __future__ import absolute_import

import collections

from mpi4py import MPI
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
