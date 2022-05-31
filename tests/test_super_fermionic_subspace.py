import pytest
import numpy as np
import src.super_fermionic_space.super_fermionic_subspace as sf_op
from scipy import sparse


# TODO: test SpinSectorDecomposition operators have stored in
# spin_sector_fermi_ops have to have the right sector

class TestClassSubspaceDecompositionSpinful:
    nsite = 2
    fl_op = sf_op.SubspaceDecomposition(nsite,
                                        tilde_conjugationrule_phase=True)

    identity_super_fermionic = sparse.eye(
        4**(fl_op.fock_ops.spin_times_site), dtype=np.complex64, format="csc")

    def test_particle_number_fock_subspace_projector_is_projector(self):
        for nelec in range(2 * self.fl_op.fock_ops.nsite + 1):
            P_sector = self.fl_op.particle_number_fock_subspace_projector(
                nelec)

            nonzeros = (P_sector * P_sector - P_sector).nnz
            assert nonzeros == 0

    def test_particle_number_fock_subspace_projector(self):
        for nelec in range(self.fl_op.fock_ops.nsite + 1):
            Pnum = self.fl_op.particle_number_fock_subspace_projector(nelec)
            N_projected = Pnum * self.fl_op.N * Pnum.transpose()

            unique, counts = np.unique(
                N_projected.diagonal(), return_counts=True)
            count = dict(zip(unique, counts))

            if nelec != 0:
                pnum_index = count[nelec] + count[0]
            else:
                pnum_index = count[0]
            assert pnum_index == self.fl_op.N.shape[0]

    def test_particle_number_fock_subspace_permutation_operator_orthogonal(
            self):
        for nelec in range(self.fl_op.fock_ops.nsite + 1):
            P_sector = \
                self.fl_op.particle_number_fock_subspace_permutation_operator(
                    nelec, full=True)

            nonzeros = (P_sector[1] * P_sector[1].transpose() -
                        self.identity_super_fermionic).nnz
            assert nonzeros == 0

    def test_particle_number_fock_subspace_permutation_operator(self):
        for nelec in range(1, self.fl_op.fock_ops.nsite + 1):
            Pnum_proj = self.fl_op.particle_number_fock_subspace_projector(
                nelec)
            Pnum_permut = (
                self.fl_op).particle_number_fock_subspace_permutation_operator(
                nelec)
            N_projected = Pnum_proj * self.fl_op.N * Pnum_proj.transpose()
            N_permut = Pnum_permut[1] * self.fl_op.N * \
                Pnum_permut[1].transpose()
            unique_N_projected, counts_N_projected = np.unique(
                N_projected.diagonal(), return_counts=True)

            unique_N_permut, counts_N_permut = np.unique(
                N_permut.diagonal(), return_counts=True)
            count_projected = dict(zip(unique_N_projected, counts_N_projected))
            count_permut = dict(zip(unique_N_permut, counts_N_permut))

            n_nelce_projected = count_projected[nelec]
            n_nelce_permut = count_permut[nelec]
            assert n_nelce_projected == n_nelce_permut

    def test_get_subspace_object(self):
        for nelec in range(1, self.fl_op.fock_ops.nsite + 1):
            Pnum_proj = self.fl_op.particle_number_fock_subspace_projector(
                nelec)
            Pnum_perm = (self.fl_op
                         ).particle_number_fock_subspace_permutation_operator(
                             nelec)
            N_projected = Pnum_proj * self.fl_op.N * Pnum_proj.transpose()
            unique, counts = np.unique(
                N_projected.diagonal(), return_counts=True)
            count = dict(zip(unique, counts))
            pnum_index = count[nelec]
            N_subspace = sf_op.get_subspace_object(self.fl_op.N, Pnum_perm,
                                                   Pnum_perm)
            assert pnum_index == N_subspace.shape[0]


class TestClassSubspaceDecompositionSpinless:
    nsite = 2
    spinless = True
    fl_op = sf_op.SubspaceDecomposition(nsite, spinless=spinless,
                                        tilde_conjugationrule_phase=True)

    identity_super_fermionic = sparse.eye(
        4**(fl_op.fock_ops.spin_times_site), dtype=np.complex64, format="csc")

    def test_particle_number_fock_subspace_projector_is_projector(self):
        for nelec in range(2 * self.fl_op.fock_ops.nsite + 1):
            P_sector = self.fl_op.particle_number_fock_subspace_projector(
                nelec)

            nonzeros = (P_sector * P_sector - P_sector).nnz
            assert nonzeros == 0

    def test_particle_number_fock_subspace_projector(self):
        for nelec in range(self.fl_op.fock_ops.nsite + 1):
            Pnum = self.fl_op.particle_number_fock_subspace_projector(nelec)
            N_projected = Pnum * self.fl_op.N * Pnum.transpose()

            unique, counts = np.unique(
                N_projected.diagonal(), return_counts=True)
            count = dict(zip(unique, counts))

            if nelec != 0:
                pnum_index = count[nelec] + count[0]
            else:
                pnum_index = count[0]
            assert pnum_index == self.fl_op.N.shape[0]

    def test_particle_number_fock_subspace_permutation_operator_orthogonal(
            self):
        for nelec in range(self.fl_op.fock_ops.nsite + 1):
            P_sector = \
                self.fl_op.particle_number_fock_subspace_permutation_operator(
                    nelec, full=True)

            nonzeros = (P_sector[1] * P_sector[1].transpose() -
                        self.identity_super_fermionic).nnz
            assert nonzeros == 0

    def test_particle_number_fock_subspace_permutation_operator(self):
        for nelec in range(1, self.fl_op.fock_ops.nsite + 1):
            Pnum_proj = self.fl_op.particle_number_fock_subspace_projector(
                nelec)
            Pnum_permut = (
                self.fl_op).particle_number_fock_subspace_permutation_operator(
                nelec)
            N_projected = Pnum_proj * self.fl_op.N * Pnum_proj.transpose()
            N_permut = Pnum_permut[1] * self.fl_op.N * \
                Pnum_permut[1].transpose()
            unique_N_projected, counts_N_projected = np.unique(
                N_projected.diagonal(), return_counts=True)

            unique_N_permut, counts_N_permut = np.unique(
                N_permut.diagonal(), return_counts=True)
            count_projected = dict(zip(unique_N_projected, counts_N_projected))
            count_permut = dict(zip(unique_N_permut, counts_N_permut))

            n_nelce_projected = count_projected[nelec]
            n_nelce_permut = count_permut[nelec]
            assert n_nelce_projected == n_nelce_permut

    def test_get_subspace_object(self):
        for nelec in range(1, self.fl_op.fock_ops.nsite + 1):
            Pnum_proj = self.fl_op.particle_number_fock_subspace_projector(
                nelec)
            Pnum_perm = (self.fl_op
                         ).particle_number_fock_subspace_permutation_operator(
                             nelec)
            N_projected = Pnum_proj * self.fl_op.N * Pnum_proj.transpose()
            unique, counts = np.unique(
                N_projected.diagonal(), return_counts=True)
            count = dict(zip(unique, counts))
            pnum_index = count[nelec]
            N_subspace = sf_op.get_subspace_object(self.fl_op.N, Pnum_perm,
                                                   Pnum_perm)
            assert pnum_index == N_subspace.shape[0]


class TestClassSpinSectorDecompositionSpinful:
    nsite = 2
    spinless = False
    fl_op = sf_op.SpinSectorDecomposition(nsite, spin_sector_max=2,
                                          spinless=spinless,
                                          tilde_conjugationrule_phase=True)

    identity_super_fermionic = sparse.eye(
        4**(fl_op.fock_ops.spin_times_site), dtype=np.complex64, format="csc")

    def test_spin_sector_projector_is_projector(self):
        for sector in self.fl_op.spin_sectors:
            P_sector = self.fl_op.spin_sector_projector(sector)

            nonzeros = (P_sector * P_sector - P_sector).nnz
            assert nonzeros == 0

    def test_spin_sector_projector_sector(self):

        for sector in self.fl_op.spin_sectors:
            P_sector = self.fl_op.spin_sector_projector(sector)
            D_up_projected = P_sector * self.fl_op.Delta_N_up \
                * P_sector.transpose()
            D_do_projected = P_sector * self.fl_op.Delta_N_do \
                * P_sector.transpose()
            sectors_projected_up = [x for x in D_up_projected.diagonal()]
            sectors_projected_do = [x for x in D_do_projected.diagonal()]

            unique_up, counts_up = np.unique(
                sectors_projected_up, return_counts=True)
            unique_do, counts_do = np.unique(
                sectors_projected_do, return_counts=True)

            count_up = dict(zip(unique_up, counts_up))
            count_do = dict(zip(unique_do, counts_do))

            if sector[0] != 0:
                p_index_up = count_up[sector[0]] + count_up[0]
            else:
                p_index_up = count_up[0]

            if sector[1] != 0:
                p_index_do = count_do[sector[1]] + count_do[0]
            else:
                p_index_do = count_do[0]
            assert p_index_up == self.fl_op.Delta_N_up.shape[0]
            assert p_index_do == self.fl_op.Delta_N_up.shape[0]

    def test_spin_sector_permutation_operator_orthogonal(self):
        for sector in self.fl_op.spin_sectors:
            P_sector = self.fl_op.spin_sector_permutation_operator(
                sector, full=True)

            nonzeros = (P_sector[1] * P_sector[1].transpose() -
                        self.identity_super_fermionic).nnz
            assert nonzeros == 0

    def test_spin_sector_permutation_operator(self):
        for sector in self.fl_op.spin_sectors:
            P_sector_permut = self.fl_op.spin_sector_permutation_operator(
                sector)

            D_up_projected = P_sector_permut[1] * self.fl_op.Delta_N_up \
                * P_sector_permut[1].transpose()
            D_do_projected = P_sector_permut[1] * self.fl_op.Delta_N_do \
                * P_sector_permut[1].transpose()

            sectors_projected_up = [x for x in D_up_projected.diagonal()]
            sectors_projected_do = [x for x in D_do_projected.diagonal()]

            unique_up, counts_up = np.unique(
                sectors_projected_up, return_counts=True)
            unique_do, counts_do = np.unique(
                sectors_projected_do, return_counts=True)

            count_up = dict(zip(unique_up, counts_up))
            count_do = dict(zip(unique_do, counts_do))

            dn_up_idx = np.where(
                self.fl_op.Delta_N_up.diagonal() == sector[0])[0]
            dn_do_idx = np.where(
                self.fl_op.Delta_N_do.diagonal() == sector[1])[0]
            mask_section = np.in1d(dn_up_idx, dn_do_idx)
            sector_idx = len(dn_up_idx[mask_section])

            if sector[0] != 0:
                assert count_up[sector[0]] == sector_idx
            else:
                assert count_up[sector[0]] == self.fl_op.Delta_N_up.shape[0]

            if sector[1] != 0:
                assert count_do[sector[1]] == sector_idx
            else:
                assert count_do[sector[1]] == self.fl_op.Delta_N_up.shape[0]


class TestClassSpinSectorDecompositionSpinless:
    nsite = 2
    spinless = True
    fl_op = sf_op.SpinSectorDecomposition(nsite, spin_sector_max=2,
                                          spinless=spinless,
                                          tilde_conjugationrule_phase=True)

    identity_super_fermionic = sparse.eye(
        4**(fl_op.fock_ops.spin_times_site), dtype=np.complex64, format="csc")

    def test_spin_sector_projector_is_projector(self):
        for sector in self.fl_op.spin_sectors:
            P_sector = self.fl_op.spin_sector_projector(sector)

            nonzeros = (P_sector * P_sector - P_sector).nnz
            assert nonzeros == 0

    def test_spin_sector_projector_sector(self):

        for sector in self.fl_op.spin_sectors:
            P_sector = self.fl_op.spin_sector_projector(sector)
            D_N_projected = P_sector * (self.fl_op.N - self.fl_op.N_tilde) \
                * P_sector.transpose()

            unique, counts = np.unique(
                D_N_projected.diagonal(), return_counts=True)

            count = dict(zip(unique, counts))

            if sector != 0:
                p_index = count[sector] + count[0]
                assert len(count) == 2
            else:
                p_index = count[0]
                assert len(count) == 1
            assert p_index == self.fl_op.N.shape[0]

    def test_spin_sector_permutation_operator_orthogonal(self):
        for sector in self.fl_op.spin_sectors:
            P_sector = self.fl_op.spin_sector_permutation_operator(
                sector, full=True)

            nonzeros = (P_sector[1] * P_sector[1].transpose() -
                        self.identity_super_fermionic).nnz
            assert nonzeros == 0

    def test_spin_sector_permutation_operator(self):
        for sector in self.fl_op.spin_sectors:
            P_sector_permut = self.fl_op.spin_sector_permutation_operator(
                sector)

            D_N_projected = P_sector_permut[1] * (
                self.fl_op.N - self.fl_op.N_tilde) \
                * P_sector_permut[1].transpose()

            sectors_projected = [x for x in D_N_projected.diagonal()]

            unique, counts = np.unique(
                sectors_projected, return_counts=True)

            count = dict(zip(unique, counts))

            dn_idx = np.where(
                (self.fl_op.N.diagonal() - self.fl_op.N_tilde.diagonal()
                 ) == sector)[0]

            sector_idx = len(dn_idx)

            if sector != 0:
                assert count[sector] == sector_idx
            else:
                assert count[sector] == self.fl_op.N.shape[0]


if __name__ == "__main__":

    pytest.main("-v test_super_fermionic_subspace.py")
