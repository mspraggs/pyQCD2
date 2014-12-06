"""Test the Lattice class"""

from __future__ import absolute_import

from mpi4py import MPI
import numpy as np
import pytest

from pyQCD2.core.lattice import (compute_halo_coords, generate_halo_sites,
                                 generate_local_sites, Lattice)


@pytest.fixture
def lattice_params():

    latshape = (8, 4, 4, 4)
    nprocs = MPI.COMM_WORLD.Get_size()
    params = {}
    params['latshape'] = np.array(latshape)
    params['latvol'] = params['latshape'].prod(0)
    params['nprocs'] = nprocs
    params['mpishape'] = np.array(MPI.Compute_dims(nprocs, len(latshape)))
    params['locshape'] = params['latshape'] // params['mpishape']
    params['locvol'] = params['locshape'].prod(0)
    params['ndims'] = len(latshape)
    params['halos'] = (params['mpishape'] > 1).astype(int)
    params['haloshape'] = params['locshape'] + 2 * params['halos']

    return params


def test_generate_local_sites():
    """Test generate_local_sites in lattice.py"""

    mpi_coord = (1, 0, 1, 1)
    local_shape = (8, 4, 4, 4)
    local_sites = generate_local_sites(np.array(mpi_coord),
                                       np.array(local_shape))
    assert local_sites.shape[0] == 2 * 4**4
    expected_sites = set([(t + 8, x, y + 4, z + 4)
                          for t in range(8) for x in range(4)
                          for y in range(4) for z in range(4)])
    local_sites_set = set([tuple(s) for s in local_sites])
    assert local_sites_set.difference(expected_sites) == set([])


def test_generate_halo_sites():
    """Test generate_halo_sites in lattice.py"""
    mpi_coord = (1, 0, 0)
    local_shape = (8, 4, 4)
    lattice_shape = (16, 8, 4)
    halos = (1, 1, 0)
    expected_sites = [(x + 8, 4, y) for x in range(8) for y in range(4)]
    expected_sites += [(x + 8, 7, y) for x in range(8) for y in range(4)]
    expected_sites += [(7, x, y) for x in range(4) for y in range(4)]
    expected_sites += [(0, x, y) for x in range(4) for y in range(4)]
    halo_sites = generate_halo_sites(np.array(mpi_coord),
                                     np.array(local_shape),
                                     np.array(lattice_shape),
                                     np.array(halos))
    halo_sites_set = set([tuple(s) for s in halo_sites])
    assert len(halo_sites) == 96
    assert halo_sites_set == set(expected_sites)


def test_compute_halo_coords():
    """Test the compute_halo_coords function in lattice.py"""
    locshape = np.array([8, 4, 4, 4])
    halos = np.array([2, 2, 2, 2])
    haloshape = np.array([10, 6, 6, 6])
    mpishape = np.array([4, 4, 4, 4])

    halo_coords = compute_halo_coords(np.array([0, 0, 0, 0]),
                                      np.array([3, 0, 0, 0]),
                                      mpishape, locshape, haloshape, halos)
    assert (halo_coords == np.array([8, 2, 2, 2])).all()
    halo_coords = compute_halo_coords(np.array([0, 0, 0, 0]),
                                      np.array([0, 0, 0, 3]),
                                      mpishape, locshape, haloshape, halos)
    assert (halo_coords == np.array([2, 2, 2, 4])).all()
    halo_coords = compute_halo_coords(np.array([4, 2, 0, 0]),
                                      np.array([0, 1, 0, 0]),
                                      mpishape, locshape, haloshape, halos)
    assert (halo_coords == np.array([6, 0, 2, 2])).all()


class TestLattice(object):

    def test_init(self, lattice_params):
        """Test the constructor"""
        lattice = Lattice((8, 4, 4, 4), 1)
        for key, value in lattice_params.items():
            if isinstance(value, np.ndarray):
                assert (getattr(lattice, key) == value).all()
            else:
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
        elif 0 in reduce(lambda x, y: x + y, lattice.mpi_neighbours):
            assert result
        else:
            assert not result

    def test_get_local_coords(self):
        """Test Lattice.get_local_coords"""
        lattice = Lattice((8, 4, 4, 4), 1)
        local_coords = lattice.get_local_coords(np.array([0, 0, 0, 0]))
        if lattice.comm.Get_rank() == 0:
            assert local_coords == tuple(map(lambda x: int(x > 1),
                                             lattice.mpishape))
        elif 0 in reduce(lambda x, y: x + y, lattice.mpi_neighbours):
            assert local_coords is not None
        else:
            assert local_coords is None
        local_coords = lattice.get_local_coords(np.array([7, 3, 3, 3]))
        if lattice.comm.Get_rank() == lattice.comm.Get_size() - 1:
            assert local_coords == tuple(map(lambda x: x[0] + x[1] - 1,
                                             zip(lattice.locshape,
                                                 lattice.halos)))
        elif (lattice.comm.Get_size() - 1
              in reduce(lambda x, y: x + y, lattice.mpi_neighbours)):
            assert local_coords is not None
        else:
            assert local_coords is None

    def test_get_local_index(self):
        """Test Lattice.get_local_index"""
        lattice = Lattice((8, 4, 4, 4), 1)
        first_index = reduce(lambda x, y: x * y[0] + y[1],
                             zip(lattice.haloshape[1:],
                                 lattice.halos[1:]), lattice.halos[0])
        local_index = lattice.get_local_index(np.array([0, 0, 0, 0]))
        if lattice.comm.Get_rank() == 0:
            assert local_index == first_index
        elif 0 in reduce(lambda x, y: x + y, lattice.mpi_neighbours):
            assert local_index is not None
        else:
            assert local_index is None
        local_index = lattice.get_local_index(np.array([7, 3, 3, 3]))
        if lattice.comm.Get_rank() == lattice.comm.Get_size() - 1:
            assert local_index == np.prod(lattice.haloshape) - first_index - 1
        elif (lattice.comm.Get_size() - 1
              in reduce(lambda x, y: x + y, lattice.mpi_neighbours)):
            assert local_index is not None
        else:
            assert local_index is None

    def test_sanitize(self):
        """Test Lattice.sanitize"""
        assert Lattice.sanitize((3, -1, 8, 2), (4, 4, 4, 4)) == (3, 3, 0, 2)