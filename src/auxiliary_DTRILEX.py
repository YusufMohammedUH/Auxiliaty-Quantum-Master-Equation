# %%
from typing import Dict
import src.super_fermionic_space.model_lindbladian as lind
import src.greens_function.frequency_greens_function as fg
import src.auxiliary_dmft as aux_dmft
import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.greens_function.correlation_functions as corr
import src.util.hdf5_util as hd5

# DT-TRILEX:
#  [X] 0. get g_aux, hyb_sys, hyb_aux
#  [X] 1. calculate auxiliary susceptibility
#  [X] 2. calculate auxiliary polarization
#  [X] 3. calculate non-interacting dual Green's function and bosonic
#         propagator
#
#  [ ] 4. calculate three vertex
#  [ ] 5. calculate gamma/delta
#  [ ] 5. calculate dual polarization
#  [ ] 7. calculate the dressed dual bosonic propagator
#  [ ] 8. calculate dual self-energy
#  [ ] 9. calculate the dressed dual Green's function
#   (optional) update self-consistently 6. with dressed dual Green's function
#              until convergence
#  [ ] 10. calculate the system Green's function


class AuxiliaryDualTRILEX:
    """This class facilitates the calculation of the system Green's function
    by using a auxiliary system as a starting point for steady state dual
    TRILEX


    Parameters
    ----------
    U_trilex : Dict
        Interaction strength for the trilex channels

    green_aux : fg.FrequencyGreen
        Auxiliary Green's function

    hyb_sys : fg.FrequencyGreen
        System hybridization function

    hyb_aux : fg.FrequencyGreen
        Auxiliary hybridization function

    correlators : corr.Correlators
        Correlators object used to calculate the vertices.

    Attributes
    ----------
    U_trilex : Dict
        Interaction strength for the trilex channels

    green_aux : fg.FrequencyGreen
        Auxiliary Green's function

    hyb_sys : fg.FrequencyGreen
        System hybridization function

    hyb_aux : fg.FrequencyGreen
        Auxiliary hybridization function

    delta_hyb : fg.FrequencyGreen
        Difference between the auxiliary and system hybridization function

    green_sys : fg.FrequencyGreen
        System Green's function

    green_bare_dual_fermion : fg.FrequencyGreen
        Bare fermionic dual Green's function

    green_dual_fermion : fg.FrequencyGreen
        Fermionic dual Green's function

    sigma_dual : fg.FrequencyGreen
        Dual fermionic self-energy

    three_point_vertex : fg.FrequencyGreen
        Dual four-point vertex

    four_point_vertex : fg.FrequencyGreen
        Dual four-point vertex

    green_bare_dual_boson : Dict
        Bare bosonic dual Green's function

    green_dual_boson : Dict
        Bosonic dual Green's function

    polarization_aux : Dict
        Auxiliary polarization

    susceptibility_aux : Dict
        Auxiliary susceptibility

    correlators : corr.Correlators
        Correlators object used to calculate the vertices.

    """

    def __init__(self, U_trilex: Dict, green_aux: fg.FrequencyGreen,
                 hyb_sys: fg.FrequencyGreen,
                 hyb_aux: fg.FrequencyGreen,
                 correlators: corr.Correlators) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        # greens functions
        self.U_trilex = U_trilex

        self.green_aux = green_aux
        self.hyb_sys = hyb_sys
        self.hyb_aux = hyb_aux
        self.delta_hyb = self.hyb_aux - self.hyb_sys

        self.green_sys = fg.FrequencyGreen(self.hyb_aux.freq)

        self.green_bare_dual_fermion = fg.FrequencyGreen(self.hyb_aux.freq)
        self.green_dual_fermion = fg.FrequencyGreen(self.hyb_aux.freq)
        self.sigma_dual = fg.FrequencyGreen(self.hyb_aux.freq)

        # For now only the following channels are implemented
        self.green_bare_dual_boson = {('ch', 'ch'): None, ('x', 'x'): None,
                                      ('x', 'y'): None, ('y', 'y'): None,
                                      ('z', 'z'): None}
        # It seams as if (x,y) and -(y,x) are the same channels
        # In paramagnetic case (z,ch) = (ch,z) = 0
        self.green_dual_boson = {
            key: None for key in self.green_bare_dual_boson}
        self.polarization_aux = {
            key: None for key in self.green_bare_dual_boson}
        self.susceptibility_aux = {
            key: None for key in self.green_bare_dual_boson}

        self.correlators = correlators
        self.three_point_vertex = {('up', 'up', 'ch'): None,
                                   ('up', 'do', 'x'): None,
                                   ('up', 'do', 'y'): None,
                                   ('up', 'up', 'z'): None}
        self.four_point_vertex = {}

    def calc_bare_dual_fermion_propagator(self) -> None:
        """Calculate the bare fermionic dual Green's function.
        """
        self.green_bare_dual_fermion = self.green_aux \
            * (self.green_aux + self.delta_hyb.inverse()).inverse() \
            * self.green_aux * (-1)

    def calc_dual_fermion_propagator(self) -> None:
        """Calculate the fermionic dual Green's function.
        """
        self.green_dual_fermion = (self.green_bare_dual_fermion.inverse()
                                   - self.sigma_dual).inverse()

    def get_system_green(self) -> None:
        """Calculate the system Green's function.
        """
        self.green_sys = self.delta_hyb.inverse() + (
            self.green_aux * self.delta_hyb).inverse() \
            * self.green_dual_fermion * (self.delta_hyb
                                         * self.green_aux).inverse()

    def calc_bare_dual_boson_propagator(self) -> None:
        """Calculate the bare bosonic dual Green's function.
        """
        for channel in self.green_bare_dual_boson:

            self.susceptibility_aux[channel] = \
                self.correlators.get_susceptibility_physical(
                self.green_aux.freq, channels=channel)

            self.polarization_aux[channel] = self.susceptibility_aux[channel] \
                * (fg.Identity +
                   self.susceptibility_aux[channel]
                    * self.U_trilex[channel[0]]).inverse()

            self.green_bare_dual_boson[channel] = (
                self.polarization_aux[channel]
                * self.U_trilex[channel[0]]) \
                * (self.susceptibility_aux[channel]
                   * self.U_trilex[channel[0]] + fg.Identity
                   * 0.5) * self.polarization_aux[channel]

    def calc_three_point_vertex(self) -> None:
        """Calculate the three-point vertex.
        """
        for spins in self.three_point_vertex:
            self.three_point_vertex[spins] = \
                self.correlators.get_three_point_vertex(
                    self.hyb_aux.freq, spin=spins, return_=True)

    def save(self, fname: str, dir_: str, dataname: str, save_parameter: bool = True,
             save_aux_data: bool = False) -> None:
        """Save the dual and auxiliary quantities to file.
        The auxilary Green's function and hybridization are passed to the
        class and therefore not saved.

        Parameters
        ----------
        fname : str
            File name to store data too.

        dir_ : str
            Group/Directory to save data too

        dataname : str
            Name under which to save T-DTRILEX data.

        save_parameter : bool, optional
            Save U_trilex if True, by default True

        save_aux_data : bool, optional
            Save auxiliary objects calculated here like auxiliary polarization
            etc., by default False
        """
        if save_parameter:
            hd5.add_attrs(fname, f"{dir_}/{dataname}", self.U_trilex)

        self.green_bare_dual_fermion.save(fname, f"{dir_}/{dataname}",
                                          "green_bare_dual_fermion",
                                          savefreq=False)
        self.green_dual_fermion.save(fname, f"{dir_}/{dataname}",
                                     "green_dual_fermion", savefreq=False)
        self.sigma_dual.save(fname, f"{dir_}/{dataname}",
                             "sigma_dual", savefreq=False)

        # TODO: saving the bare propagators should be optional
        for channel in self.green_bare_dual_boson:
            self.green_bare_dual_boson[channel].save(
                fname, f"{dir_}/{dataname}/green_bare_dual_boson",
                f"{channel}", savefreq=False)

            self.green_dual_boson[channel].save(
                fname, f"{dir_}/{dataname}/green_dual_boson",
                f"{channel}", savefreq=False)
            if save_aux_data:
                self.polarization_aux[channel].save(
                    fname, f"{dir_}/{dataname}/polarization_aux", f"{channel}",
                    savefreq=False)

                self.susceptibility_aux[channel].save(
                    fname, f"{dir_}/{dataname}/susceptibility_aux",
                    f"{channel}", savefreq=False)

        hd5.add_dict_data(fname, f"{dir_}/{dataname}", 'three_point_vertex',
                          self.three_point_vertex)
        hd5.add_dict_data(fname, f"{dir_}/{dataname}", 'four_point_vertex',
                          self.four_point_vertex)

    def load(self, fname: str, dir_: str, dataname: str,
             load_parameter: bool = True, load_aux_data: bool = False) -> None:
        """Load the dual and auxiliary quantities from file.
        The auxilary Green's function and hybridization are passed to the
        class and therefore not saved.

        Parameters
        ----------
        fname : str
            File name to store data too.

        dir_ : str
            Group/Directory to save data too.

        dataname : str
            Name under which to save T-DTRILEX data.

        load_parameter : bool, optional
            Load U_trilex if True, by default True

        load_aux_data : bool, optional
            Load auxiliary objects calculated here like auxiliary polarization
            etc., by default False
        """
        if load_aux_data:
            self.U_trilex = hd5.read_attrs(fname, f"{dir_}/{dataname}")

        self.green_bare_dual_fermion.load(fname, f"{dir_}/{dataname}",
                                          "green_bare_dual_fermion",
                                          readfreq=False)
        self.green_dual_fermion.load(fname, f"{dir_}/{dataname}",
                                     "green_dual_fermion", readfreq=False)
        self.sigma_dual.load(fname, f"{dir_}/{dataname}",
                             "sigma_dual", readfreq=False)

        # TODO: saving the bare propagators should be optional
        for channel in self.green_bare_dual_boson:
            self.green_bare_dual_boson[channel].load(
                fname, f"{dir_}/{dataname}/green_bare_dual_boson",
                f"{channel}", readfreq=False)

            self.green_dual_boson[channel].load(
                fname, f"{dir_}/{dataname}/green_dual_boson",
                f"{channel}", readfreq=False)
            if load_aux_data:
                self.polarization_aux[channel].load(
                    fname, f"{dir_}/{dataname}/polarization_aux", f"{channel}",
                    readfreq=False)

                self.susceptibility_aux[channel].load(
                    fname, f"{dir_}/{dataname}/susceptibility_aux",
                    f"{channel}", readfreq=False)

        hd5.read_dict_data(fname, f"{dir_}/{dataname}", 'three_point_vertex')
        hd5.read_dict_data(fname, f"{dir_}/{dataname}", 'four_point_vertex')


if __name__ == '__main__':
    #  Frequency grid
    N_freq = 400
    freq_max = 10

    selfconsist_param = {'max_iter': 50, 'err_tol': 1e-7, 'mixing': 0.2}

    e0 = 0
    mu = 0
    beta = 100
    D = 11
    gamma = 0.1

    leads_param = {'e0': e0, 'mu': [mu], 'beta': beta, 'D': D, 'gamma': gamma}

    spinless = False
    spin_sector_max = 2
    tilde_conjugationrule_phase = True

    U = 3.0
    v = 1.0
    sys_param = {'v': v, 'U': U, 'spinless': spinless,
                 'tilde_conjugation': tilde_conjugationrule_phase}

    # Parameters of the auxiliary system
    Nb = 1
    nsite = 2 * Nb + 1
    aux_param = {'Nb': Nb, 'nsite': nsite}
    trilex = {'ch': U / 2., 'x': -U / 2, 'y': -U / 2, 'z': -U / 2}
    params = {'freq': {"freq_min": -freq_max, "freq_max": freq_max,
                       'N_freq': N_freq},
              'selfconsistency': selfconsist_param, 'leads': leads_param,
              'aux_sys': aux_param, 'system': sys_param, 'U_trilex': trilex}

    # ##################### Initializing Lindblad class #######################
    super_fermi_ops = sf_sub.SpinSectorDecomposition(
        nsite, spin_sector_max, spinless=spinless,
        tilde_conjugationrule_phase=tilde_conjugationrule_phase)

    L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)
    corr_cls = corr.Correlators(L, trilex=True)
    auxiliaryDMFT = aux_dmft.AuxiliaryMaserEquationDMFT(
        params, correlators=corr_cls)
# %%
if __name__ == '__main__':
    auxiliaryDMFT.hyb_leads = auxiliaryDMFT.get_bath()
    auxiliaryDMFT.set_local_matrix()
    auxiliaryDMFT.solve()
    auxiliaryDMFT.save('ForAuxTrilex.h5')

# %%
if __name__ == '__main__':
    auxiliaryDMFT.load('ForAuxTrilex.h5', read_parameters=True)
    auxTrilex = AuxiliaryDualTRILEX(U_trilex=params['U_trilex'],
                                    green_aux=auxiliaryDMFT.green_aux,
                                    hyb_sys=(auxiliaryDMFT.hyb_leads
                                    + auxiliaryDMFT.hyb_dmft),
                                    hyb_aux=auxiliaryDMFT.hyb_aux,
                                    correlators=corr_cls)
    auxTrilex.calc_bare_dual_fermion_propagator()
    auxTrilex.calc_dual_fermion_propagator()
    auxTrilex.get_system_green()
    auxTrilex.calc_three_point_vertex()
    auxTrilex.calc_four_point_vertex()
    auxTrilex.save(fname='ForAuxTrilex.h5', dir_='/', dataname='trilex',
                   save_aux_data=False)
# %%
