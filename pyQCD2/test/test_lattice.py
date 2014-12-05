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
    params['locvol'] = reduce(lambda x, y: x * y,
                              params['locshape'])
    params['ndims'] = len(latshape)
    params['halos'] = map(lambda x: int(x > 1), params['mpishape'])
    params['haloshape'] = tuple([x // y + 2 * z
                                 for x, y, z in zip(params['latshape'],
                                                    params['mpishape'],
                                                    params['halos'])])

    return params


class TestLattice(object):

    def test_init(self, lattice_params):
        """Test the constructor"""
        lattice = Lattice((8, 4, 4, 4), 1)
        for key, value in lattice_params.items():
            assert getattr(lattice, key) == value
        assert isinstance(lattice.comm, MPI.Cartcomm)
        assert len(lattice.local_sites) == lattice_params['locvol']
        my_coord = lattice.comm.Get_coords(lattice.comm.Get_rank())
        # MPI neighbours shouldn't be more than 1 hop away
        for axis_neighbours in lattice.mpi_neighbours:
            for neighbour in axis_neighbours:
                neighbour_coord = lattice.comm.Get_coords(neighbour)
                diffs = map(lambda x: abs(x[1] - x[0]),
                            zip(my_coord, neighbour_coord))
                # Account for periodic boundary conditions
                for i, diff in enumerate(diffs):
                    if diff > lattice.mpishape[i] / 2:
                        diffs[i] = lattice.mpishape[i] - diff
                assert sum(diffs) == 1
        assert len(lattice.mpi_neighbours) == 4

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
        local_coords = lattice.get_local_coords((0, 0, 0, 0))
        if lattice.comm.Get_rank() == 0:
            assert local_coords == tuple(map(lambda x: int(x > 1),
                                             lattice.mpishape))
        else:
            assert local_coords is None
        local_coords = lattice.get_local_coords((7, 3, 3, 3))
        if lattice.comm.Get_rank() == lattice.comm.Get_size() - 1:
            assert local_coords == tuple(map(lambda x: x[0] + x[1] - 1,
                                             zip(lattice.locshape,
                                                 lattice.halos)))
        else:
            assert local_coords is None

    def test_get_local_index(self):
        """Test Lattice.get_local_index"""
        lattice = Lattice((8, 4, 4, 4), 1)
        first_index = reduce(lambda x, y: x * y[0] + y[1],
                             zip(lattice.haloshape[1:],
                                 lattice.halos[1:]), lattice.halos[0])
        local_index = lattice.get_local_index((0, 0, 0, 0))
        if lattice.comm.Get_rank() == 0:
            assert local_index == first_index
        else:
            assert local_index is None
        local_index = lattice.get_local_index((7, 3, 3, 3))
        if lattice.comm.Get_rank() == lattice.comm.Get_size() - 1:
            assert local_index == np.prod(lattice.haloshape) - first_index - 1
        else:
            assert local_index is None

    def test_sanitize(self):
        """Test Lattice.sanitize"""
        assert Lattice.sanitize((3, -1, 8, 2), (4, 4, 4, 4)) == (3, 3, 0, 2)