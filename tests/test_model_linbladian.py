
import pytest
import numpy as np
import src.super_fermionic_subspace as sop_sub
import src.model_lindbladian as lind
import src.model_hamiltonian as ham

# TODO: add more tests


class TestClassLindbladianSpinSectorDecomposition:
    nsite = 2
    sector_max = 2
    spinless = False
    tilde_conjugationrule_phase = True
    super_fermionic_op = sop_sub.SpinSectorDecomposition(
        nsite=nsite, spin_sector_max=sector_max, spinless=spinless,
        tilde_conjugationrule_phase=tilde_conjugationrule_phase)
    Lindblad = lind.Lindbladian(super_fermi_ops=super_fermionic_op)

    super_fermionic_op.fock_ops.spinless
    es = np.ones(nsite)
    ts = np.ones(nsite - 1)
    Gamma1 = 0.1 * np.ones((nsite, nsite))
    Gamma2 = 0.1 * np.ones((nsite, nsite))

    T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
        nsite=nsite, es=es, ts=ts)
    U_mat = 0.2 * np.ones(nsite)
    Lindblad.update(T_mat=T_mat, U_mat=U_mat, Gamma1=Gamma1,
                    Gamma2=Gamma2)

    def test_consturction_unitary(self):
        real = self.Lindblad.L_unitary + self.Lindblad.L_unitary.conj()
        assert (real).nnz == 0
        assert (self.Lindblad.L_unitary
                - self.Lindblad.L_unitary.transpose()).nnz == 0

    def test_consturction_dissipation(self):
        L_D1 = self.Lindblad.L_Gamma1
        L_D2 = self.Lindblad.L_Gamma2

        L_D1_real = L_D1 + L_D1.conj()
        L_D2_real = L_D2 + L_D2.conj()

        L_D1_imag = L_D1 - L_D1.conj()
        L_D2_imag = L_D2 - L_D2.conj()

        assert (L_D1_real).nnz != 0
        assert (L_D2_real).nnz != 0

        assert (L_D1_imag).nnz != 0
        assert (L_D2_imag).nnz != 0


class TestClassLindbladianSpinSectorDecompositionNoComplexPhase:
    nsite = 2
    sector_max = 2
    spinless = False
    tilde_conjugationrule_phase = False
    super_fermionic_op = sop_sub.SpinSectorDecomposition(
        nsite=nsite, spin_sector_max=sector_max, spinless=spinless,
        tilde_conjugationrule_phase=tilde_conjugationrule_phase)
    Lindblad = lind.Lindbladian(super_fermi_ops=super_fermionic_op)

    super_fermionic_op.fock_ops.spinless
    es = np.ones(nsite)
    ts = np.ones(nsite - 1)
    Gamma1 = 0.1 * np.ones((nsite, nsite))
    Gamma2 = 0.1 * np.ones((nsite, nsite))

    T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
        nsite=nsite, es=es, ts=ts)
    U_mat = 0.2 * np.ones(nsite)
    Lindblad.update(T_mat=T_mat, U_mat=U_mat, Gamma1=Gamma1,
                    Gamma2=Gamma2)

    def test_consturction_unitary(self):
        real = self.Lindblad.L_unitary + self.Lindblad.L_unitary.conj()
        assert (real).nnz == 0
        assert (self.Lindblad.L_unitary
                - self.Lindblad.L_unitary.transpose()).nnz == 0

    def test_consturction_dissipation(self):
        L_D1 = self.Lindblad.L_Gamma1
        L_D2 = self.Lindblad.L_Gamma2

        L_D1_real = L_D1 + L_D1.conj()
        L_D2_real = L_D2 + L_D2.conj()

        L_D1_imag = L_D1 - L_D1.conj()
        L_D2_imag = L_D2 - L_D2.conj()

        assert (L_D1_real).nnz != 0
        assert (L_D2_real).nnz != 0

        assert (L_D1_imag).nnz == 0
        assert (L_D2_imag).nnz == 0


if __name__ == "__main__":

    pytest.main("-v test_model_lindbladian.py")
