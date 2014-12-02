"""Test the Lattice class"""

from __future__ import absolute_import

from mpi4py import MPI
import numpy as np
import pytest

from pyQCD2.core.lattice import Lattice


@pytest.fixture
def lattice_params():

    latshape = (8, 4, 4, 4)
    nprocs = MPI.COMM_WORLD.Get_size()
    params = {}
    params['latshape'] = latshape
    params['latvol'] = reduce(lambda x, y: x * y,
                              params['latshape'])
    params['nprocs'] = nprocs
    params['mpishape'] = tuple(MPI.Compute_dims(nprocs, len(latshape)))
    params['locshape'] = tuple([x // y for x, y in zip(params['latshape'],
                                                       params['mpishape'])])
    params['datashape'] = tuple([x // y + 2 for x, y in zip(params['latshape'],
                                                            params['mpishape'])])
    params['locvol'] = reduce(lambda x, y: x * y,
                              params['locshape'])
    params['ndims'] = len(latshape)
    params['halo'] = 1

    return params


class TestLattice(object):

    def test_init(self, lattice_params):
        """Test the constructor"""
        # TODO: Test in here for neighbour ranks
        lattice = Lattice((8, 4, 4, 4), 1)
        for key, value in lattice_params.items():
            assert getattr(lattice, key) == value
        assert isinstance(lattice.comm, MPI.Cartcomm)
        assert len(lattice.sites) == lattice_params['locvol']

        if lattice_params['nprocs'] > 1:
            with pytest.raises(RuntimeError):
                nprocs = lattice_params['nprocs']
                Lattice((nprocs + 1, nprocs, nprocs, nprocs))

    def test_ishere(self):
        """Test Lattice.ishere"""
        lattice = Lattice((8, 4, 4, 4), 1)
        result = lattice.ishere((0, 0, 0, 0))
        if lattice.comm.Get_rank() == 0:
            assert result
        else:
            assert not result

    def test_get_local_coords(self):
        """Test Lattice.get_local_coords"""
        lattice = Lattice((8, 4, 4, 4), 1)
        assert lattice.get_local_coords((0, 0, 0, 0)) == (1, 1, 1, 1)
        assert lattice.get_local_coords((7, 3, 3, 3)) == lattice.locshape

    def test_get_local_index(self):
        """Test Lattice.get_local_index"""
        lattice = Lattice((8, 4, 4, 4), 1)
        first_index = reduce(lambda x, y: x * y[0] + y[1],
                             zip(lattice.datashape[-1:0:-1],
                                 (1, 1, 1)), 1)
        assert lattice.get_local_index((0, 0, 0, 0)) == first_index
        assert (lattice.get_local_index((7, 3, 3, 3))
                == np.prod(lattice.data_shape) - first_index - 1)