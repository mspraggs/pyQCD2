"""Test the Lattice class"""

from __future__ import absolute_import

from mpi4py import MPI
import numpy as np
import pytest

from pyQCD2.core.lattice import (coord_to_index, compute_halo_coords,
                                 compute_neighbours, generate_halo_sites,
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

    def not_in_corner(w, z):
        return (w, z) not in [(-1, -1), (-1, 4), (8, -1), (8, 4)]

    loc_sites = set([tuple(map(sum, zip(n, (8, 0, 0))))
                     for n in np.ndindex(local_shape)])
    for max_hops, condition in zip([3, 1], [lambda t, x: True,
                                            not_in_corner]):
        all_sites = set([((t + 8) % 16, x % 8, y % 4)
                         for t in range(-1, 9)
                         for x in range(-1, 5)
                         for y in range(4) if condition(t, x)])
        expected_sites = all_sites.difference(loc_sites)
        halo_sites = generate_halo_sites(np.array(mpi_coord),
                                         np.array(local_shape),
                                         np.array(lattice_shape),
                                         np.array(halos), max_hops)
        halo_sites_set = set([tuple(s) for s in halo_sites])
        assert len(halo_sites) == len(expected_sites)
        assert halo_sites_set == expected_sites


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
    halo_coords = compute_halo_coords(np.array([0, 0, 0, 0]),
                                      np.array([3, 3, 3, 3]),
                                      mpishape, locshape, haloshape, halos)
    assert (halo_coords == np.array([8, 4, 4, 4])).all()


def test_compute_neighbours():
    """Test the comptue_neighbours function in lattice.py"""
    mpishape = np.array([2, 1, 1, 1])
    mpicoord = np.array([0, 0, 0, 0])
    locshape = np.array([8, 4, 8, 6])
    halos = np.array([2, 0, 0, 0])
    neighbours_data = compute_neighbours(mpicoord, mpishape, locshape,
                                         halos, -1)
    assert (neighbours_data[0] == np.array([[1, 0, 0, 0]])).all()
    assert (neighbours_data[0] == neighbours_data[1]).all()
    assert (neighbours_data[2] == -neighbours_data[3]).all()
    assert (neighbours_data[4] == np.array([[2, 4, 8, 6]])).all()

    mpishape = np.array([2, 1, 2, 1])
    mpicoord = np.array([0, 0, 1, 0])
    locshape = np.array([8, 4, 4, 6])
    halos = np.array([2, 0, 2, 0])
    neighbours_data = compute_neighbours(mpicoord, mpishape, locshape,
                                         halos, -1)
    assert (neighbours_data[0] == np.array([[0, 0, 0, 0],
                                            [1, 0, 0, 0],
                                            [1, 0, 1, 0],
                                            [1, 0, 0, 0]])).all()
    assert (neighbours_data[0] == neighbours_data[1]).all()
    assert (np.abs(neighbours_data[2]).sum(axis=1) <= 3).all()
    assert (np.abs(neighbours_data[2]) <= 1).all()
    assert (neighbours_data[2] == -neighbours_data[3]).all()
    assert (neighbours_data[4] == np.array([[8, 4, 2, 6],
                                            [2, 4, 2, 6],
                                            [2, 4, 4, 6],
                                            [2, 4, 2, 6]])).all()


def test_coord_to_index():
    """Test coord_to_index in lattice.py"""
    assert coord_to_index((1, 3, 2, 1), (8, 4, 5, 6)) == 223
    assert coord_to_index((0, 0, 0, 0), (8, 4, 5, 6)) == 0
    assert coord_to_index((7, 3, 4, 5), (8, 4, 5, 6)) == 959


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
        assert lattice.local_site_coords.shape[0] == lattice_params['locvol']
        assert lattice.local_site_indices.size == lattice_params['locvol']

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
        elif 0 in lattice.fnt_neighb_ranks or 0 in lattice.bck_neighb_ranks:
            assert result
        else:
            assert not result

    def test_halo_slice(self):
        """Test the halo slice specification function"""
        lattice = Lattice((8, 4, 4, 4), 1)
        slicer = lattice.halo_slice(np.array([0, 0, -1, 0]), 'send')
        expected_slicer = [slice(h, -h) if h > 0 else slice(None)
                           for h in lattice.halos]
        expected_slicer[2] = slice(lattice.halos[2], 2 * lattice.halos[2])
        assert tuple(expected_slicer) == slicer
        slicer = lattice.halo_slice(np.array([0, 1, 0, 0]), 'recv')
        expected_slicer = [slice(h, -h) if h > 0 else slice(None)
                           for h in lattice.halos]
        expected_slicer[1] = slice(-lattice.halos[1], None)
        assert tuple(expected_slicer) == slicer
        slicer = lattice.halo_slice(np.array([-1, 1, 0, 0]), 'recv')
        expected_slicer = [slice(h, -h) if h > 0 else slice(None)
                           for h in lattice.halos]
        expected_slicer[0] = slice(None, lattice.halos[0])
        expected_slicer[1] = slice(-lattice.halos[1], None)
        assert tuple(expected_slicer) == slicer

    def test_get_site_rank(self):
        """Test the get_site_rank function"""
        lattice = Lattice((8, 4, 4, 4), 1)
        assert lattice.get_site_rank((0, 0, 0, 0)) == 0
        assert (lattice.get_site_rank((7, 3, 3, 3))
                == lattice.comm.Get_size() - 1)

    def test_get_local_coords(self):
        """Test Lattice.get_local_coords"""
        lattice = Lattice((8, 4, 4, 4), 1)
        local_coords = lattice.get_local_coords(np.array([0, 0, 0, 0]))
        if lattice.comm.Get_rank() == 0:
            assert local_coords == tuple([int(x > 1) for x in lattice.mpishape])
        elif 0 in lattice.fnt_neighb_ranks or 0 in lattice.bck_neighb_ranks:
            assert local_coords is not None
        else:
            assert local_coords is None
        local_coords = lattice.get_local_coords(np.array([7, 3, 3, 3]))
        if lattice.comm.Get_rank() == lattice.comm.Get_size() - 1:
            assert local_coords == tuple([x + y - 1
                                          for x, y in zip(lattice.locshape,
                                                          lattice.halos)])
        elif (lattice.comm.Get_size() - 1
              in np.append(lattice.fnt_neighb_ranks, lattice.bck_neighb_ranks)):
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
        elif 0 in lattice.bck_neighb_ranks or 0 in lattice.fnt_neighb_ranks:
            assert local_index is not None
        else:
            assert local_index is None
        local_index = lattice.get_local_index(np.array([7, 3, 3, 3]))
        if lattice.comm.Get_rank() == lattice.comm.Get_size() - 1:
            assert local_index == np.prod(lattice.haloshape) - first_index - 1
        elif (lattice.comm.Get_size() - 1
              in np.append(lattice.bck_neighb_ranks, lattice.fnt_neighb_ranks)):
            assert local_index is not None
        else:
            assert local_index is None

    def test_sanitize(self):
        """Test Lattice.sanitize"""
        assert Lattice.sanitize((3, -1, 8, 2), (4, 4, 4, 4)) == (3, 3, 0, 2)
