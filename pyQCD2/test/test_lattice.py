"""Test the Lattice class"""

from __future__ import absolute_import

from mpi4py import MPI
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
    params['locvol'] = reduce(lambda x, y: x * y,
                              params['locshape'])
    params['ndims'] = len(latshape)
    params['halo'] = 1

    return params


class TestLattice(object):

    def test_init(self, lattice_params):
        """Test the constructor"""
        lattice = Lattice((8, 4, 4, 4), 1)
        for key, value in lattice_params.items():
            assert getattr(lattice, key) == value
        assert isinstance(lattice.comm, MPI.Cartcomm)
        assert len(lattice.sites) == lattice_params['locvol']

        if lattice_params['nprocs'] > 1:
            with pytest.raises(RuntimeError):
                nprocs = lattice_params['nprocs']
                Lattice((nprocs + 1, nprocs, nprocs, nprocs))