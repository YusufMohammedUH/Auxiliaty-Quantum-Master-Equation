# %%
from typing import Dict, Optional
import numpy as np
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
    filename : Optional[str], optional
        File name in which data is stored, by default None

    dir_ : Optional[str], optional
        HDF5 group/ subdirectory in which data is stored, by default None

    dataname : Optional[str], optional
        Name under which data is stored, by default None

    load_input_param : bool, optional
        Load input parameters like green_aux, hyb_sys, etc. if True, by default
        True.

    load_aux_data : bool, optional
        Load auxiliary objects calculated here like auxiliary polarization
        etc. if True, by default False

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

    Raises
    ------
    ValueError
        'Either all or none of the following arguments must be given: filename,
        dir_, dataname. If nonn are given, U_trilex, green_aux, hyb_sys,
        hyb_aux and correlators are required.'
    """

    def __init__(self, *, filename: Optional[str] = None,
                 dir_: Optional[str] = None,
                 dataname: Optional[str] = None,
                 load_input_param: bool = True,
                 load_aux_data: bool = False,
                 U_trilex: Optional[Dict] = None,
                 green_aux: Optional[fg.FrequencyGreen] = None,
                 hyb_sys: Optional[fg.FrequencyGreen] = None,
                 hyb_aux: Optional[fg.FrequencyGreen] = None,
                 correlators: Optional[corr.Correlators] = None) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        self.correlators = correlators
        if (filename is None) and (dir_ is None) and (dataname is None):
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
            self.green_bare_dual_boson = {
                ('ch', 'ch'): fg.FrequencyGreen(self.hyb_aux.freq),
                ('x', 'x'): fg.FrequencyGreen(self.hyb_aux.freq),
                ('x', 'y'): fg.FrequencyGreen(self.hyb_aux.freq),
                ('y', 'y'): fg.FrequencyGreen(self.hyb_aux.freq),
                ('z', 'z'): fg.FrequencyGreen(self.hyb_aux.freq)}
            # It seams as if (x,y) and -(y,x) are the same channels
            # In paramagnetic case (z,ch) = (ch,z) = 0
            self.green_dual_boson = {
                key: fg.FrequencyGreen(self.hyb_aux.freq) for key in
                self.green_bare_dual_boson}
            self.polarization_aux = {
                key: fg.FrequencyGreen(self.hyb_aux.freq) for key in
                self.green_bare_dual_boson}
            self.susceptibility_aux = {
                key: fg.FrequencyGreen(self.hyb_aux.freq) for key in
                self.green_bare_dual_boson}

            self.three_point_vertex = {('up', 'up', 'ch'): None,
                                       ('up', 'do', 'x'): None,
                                       ('up', 'do', 'y'): None,
                                       ('up', 'up', 'z'): None}
            self.four_point_vertex = {}
        elif (filename is not None) and (
                dir_ is not None) and (dataname is not None):
            self.load(fname=filename, dir_=dir_, dataname=dataname,
                      load_input_param=load_input_param,
                      load_aux_data=load_aux_data)
        else:
            raise ValueError('Either all or none of the following arguments ' +
                             'must be given: filename, dir_, dataname. ' +
                             'If nonn are given, U_trilex, green_aux,' +
                             ' hyb_sys, hyb_aux and ' +
                             'correlators are required.')

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
            print(f"Vertex with spins {spins} calculated")

    def save(self, fname: str, dir_: str, dataname: str,
             save_input_param: bool = True, save_aux_data: bool = False) -> None:
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

         save_input_param : str
            Save input parameters like green_aux, hyb_sys, etc, by default True.

        save_aux_data : bool, optional
            Save auxiliary objects calculated here like auxiliary polarization
            etc., by default False
        """
        hd5.add_attrs(fname, f"{dir_}/{dataname}", self.U_trilex)

        if save_input_param:
            freq = {'freq_min': self.hyb_aux.freq[0],
                    'freq_max': self.hyb_aux.freq[-1],
                    'N_freq': len(self.hyb_aux.freq)}
            hd5.add_attrs(fname, "/", freq)
            self.green_aux.save(fname, '/auxiliary_sys',
                                'green_aux', savefreq=False)
            self.hyb_aux.save(fname, '/auxiliary_sys', 'hyb_aux',
                              savefreq=False)
            if ('hyb_dmft' not in hd5.get_directorys(fname, '/system') or
                    'hyb_leads' not in hd5.get_directorys(fname, '/system')):
                self.hyb_sys.save(fname, '/system', 'hyb_sys', savefreq=False)

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

            # self.green_dual_boson[channel].save(
            #     fname, f"{dir_}/{dataname}/green_dual_boson",
            #     f"{channel}", savefreq=False)
            if save_aux_data:
                self.polarization_aux[channel].save(
                    fname, f"{dir_}/{dataname}/polarization_aux", f"{channel}",
                    savefreq=False)

                self.susceptibility_aux[channel].save(
                    fname, f"{dir_}/{dataname}/susceptibility_aux",
                    f"{channel}", savefreq=False)

        hd5.add_dict_data(fname, f"{dir_}/{dataname}", 'three_point_vertex',
                          self.three_point_vertex)
        # hd5.add_dict_data(fname, f"{dir_}/{dataname}", 'four_point_vertex',
        #                   self.four_point_vertex)

    def load(self, fname: str, dir_: str, dataname: str,
             load_input_param: bool = True, load_aux_data: bool = False) -> None:
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

        load_input_param : bool, optional
            Load input parameters like green_aux, hyb_sys, etc, by default
            True.

        load_aux_data : bool, optional
            Load auxiliary objects calculated here like auxiliary polarization
            etc., by default False
        """
        self.U_trilex = hd5.read_attrs(fname, f"{dir_}/{dataname}")
        if self.correlators is None:
            spin_sector_max = 2
            aux_param = hd5.read_attrs(fname, '/auxiliary_sys')
            sys_param = hd5.read_attrs(fname, '/system')
            super_fermi_ops = sf_sub.SpinSectorDecomposition(
                nsite=aux_param['nsite'], spin_sector_max=spin_sector_max,
                spinless=sys_param['spinless'],
                tilde_conjugationrule_phase=sys_param['tilde_conjugation'])

            L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)
            self.correlators = corr.Correlators(L, trilex=True)

        self.correlators.Lindbladian.load(fname, '/auxiliary_sys')
        self.correlators.update(T_mat=self.correlators.Lindbladian.T_mat,
                                U_mat=self.correlators.Lindbladian.U_mat,
                                Gamma1=self.correlators.Lindbladian.Gamma1,
                                Gamma2=self.correlators.Lindbladian.Gamma2)
        if load_input_param:
            freq_parm = hd5.read_attrs(fname, '/')

            freq = np.linspace(freq_parm['freq_min'], freq_parm['freq_max'],
                               freq_parm['N_freq'])

            self.green_aux = fg.FrequencyGreen(freq)
            self.green_aux.load(fname, '/auxiliary_sys',
                                'green_aux', readfreq=False)

            self.hyb_aux = fg.FrequencyGreen(freq)
            self.hyb_aux.load(fname, '/auxiliary_sys',
                              'hyb_aux', readfreq=False)

            if ('hyb_dmft' in hd5.get_directorys(fname, '/system') and
                    'hyb_leads' in hd5.get_directorys(fname, '/system')):
                hyb_dmft = fg.FrequencyGreen(freq)
                hyb_leads = fg.FrequencyGreen(freq)

                hyb_dmft.load(fname, '/system', 'hyb_dmft', readfreq=False)
                if 'hyb_leads' in hd5.get_directorys(fname, '/system'):
                    hyb_leads.load(fname, '/system',
                                   'hyb_leads', readfreq=False)

                self.hyb_sys = hyb_dmft + hyb_leads
            else:
                self.hyb_sys.load(fname, '/system', 'hyb_sys', readfreq=False)

            self.__init__(U_trilex=self.U_trilex, green_aux=self.green_aux,
                          hyb_sys=self.hyb_sys, hyb_aux=self.hyb_aux,
                          correlators=self.correlators)

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

            # self.green_dual_boson[channel].load(
            #     fname, f"{dir_}/{dataname}/green_dual_boson",
            #     f"{channel}", readfreq=False)

            if load_aux_data:
                self.polarization_aux[channel].load(
                    fname, f"{dir_}/{dataname}/polarization_aux", f"{channel}",
                    readfreq=False)

                self.susceptibility_aux[channel].load(
                    fname, f"{dir_}/{dataname}/susceptibility_aux",
                    f"{channel}", readfreq=False)

        temp = hd5.read_dict_data(
            fname, f"{dir_}/{dataname}", 'three_point_vertex')
        for key in self.three_point_vertex:
            self.three_point_vertex[key] = temp[f"{key}"]

        # temp = hd5.read_dict_data(
        #     fname, f"{dir_}/{dataname}", 'four_point_vertex')
        # for key in self.four_point_vertex.keys():
        #     self.four_point_vertex[key] = temp[f"{key}"]


if __name__ == '__main__':
    auxTrilex = AuxiliaryDualTRILEX(
        filename='ForAuxTrilex.h5', dir_='', dataname='trilex')

    auxTrilex.calc_bare_dual_boson_propagator()
# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import src.util.figure as fu
    import colorcet as cc
    plot = fu.FigureTheme(cmap=cc.cm.bgy)

    plot.create_single_panel(projection_='3d',
                             xlabel='t\'',
                             ylabel='t',
                             zlabel='$G^{R}$')
    plot.ax.imshow(auxTrilex.three_point_vertex[
        ('up', 'up', 'ch')][:, :, 0, 0, 0].imag, cmap=cc.cm.bgy)
    # plot.surface(auxTrilex.green_aux.freq, auxTrilex.green_aux.freq,
    #              auxTrilex.three_point_vertex[('up', 'up', 'ch')
    #                                           ][:, :, 1, 1,
    #                                             0].imag)
    plt.show()
# %%
