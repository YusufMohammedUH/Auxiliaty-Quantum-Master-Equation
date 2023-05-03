# %%
from typing import Dict, Optional
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import src.greens_function.frequency_greens_function as fg
import src.greens_function.correlation_functions as corr
import src.util.hdf5_util as hd5
import src.greens_function.dos_util as du
import src.util.fourier as dft
import src.impurity_solver.auxiliary_solver_base as aux_base
# import src.dmft.auxiliary_dmft as aux_dmft

import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.super_fermionic_space.model_lindbladian as lind
# Dual TRILEX:
#  [X] 1. inherit from AuxiliaryDualSolverBase:
#  [X] 2. copy from AuxiliaryDualTRILEX
#  [X] 3. calculate the three point correlator
#  [X] 4. calculate convolution of three point correlator with green's function
#         and obtain three point vertex
#  [X] 5. calculate "mirrored" three point vertex
#  [ ] 6. calculate polarization:
#           [ ]- first derive convolution for the polarization
#           [ ]- then calculate the polarization
#  [ ] 7. calculate self-energy
#           [ ]- first derive convolution for the self-energy
#           [ ]- then calculate the self-energy


class AuxiliaryDualTRILEX(aux_base.AuxiliaryDualSolverBase):
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

    green_bare_dual : fg.FrequencyGreen
        Bare fermionic dual Green's function

    green_dual : fg.FrequencyGreen
        Fermionic dual Green's function

    sigma_dual : fg.FrequencyGreen
        Dual fermionic self-energy

    bare_dual_screened_interaction : Dict
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
                 time_param: Optional[Dict] = None,
                 U_trilex: Optional[Dict] = None,
                 green_aux: Optional[fg.FrequencyGreen] = None,
                 hyb_sys: Optional[fg.FrequencyGreen] = None,
                 hyb_aux: Optional[fg.FrequencyGreen] = None,
                 correlators: Optional[corr.Correlators] = None,
                 keldysh_comp: str = 'keldysh') -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        self.time = np.linspace(time_param['time_min'],
                                time_param['time_max'],
                                time_param['N_time'])
        self.contour_components = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                                   (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

        self.three_point_vertex = {('up', 'up', 'ch'): None,
                                   ('up', 'do', 'x'): None,
                                   ('up', 'do', 'y'): None,
                                   ('up', 'up', 'z'): None}
        self.four_point_vertex = {}

        super().__init__(filename=filename, dir_=dir_, dataname=dataname,
                         load_input_param=load_input_param,
                         load_aux_data=load_aux_data,
                         U_trilex=U_trilex,
                         green_aux=green_aux,
                         hyb_sys=hyb_sys,
                         hyb_aux=hyb_aux,
                         correlators=correlators,
                         keldysh_comp=keldysh_comp)

    def compute_three_point_correlator(self) -> None:
        """Calculate the three point vertex
        """
        for spins in self.three_point_vertex:
            self.three_point_correlator[spins] = \
                self.correlators.get_three_point_vertex(
                freq=self.green_aux.freq, spin=spins, return_=True)

    def get_three_point_vertex(self) -> None:
        """Calculate the three point vertex, by convoluting the green's
        functions and susceptibilities.
        """
        for con_comp in self.contour_components:
            for spins in self.three_point_vertex:
                self.three_point_vertex = \
                    self.correlators.get_vertex_green_convolution(
                        green=self.green_aux.inverse(), component=con_comp,
                        spin=spins, position_freq=(0, 0),
                        vertex=self.three_point_vertex)
                self.three_point_vertex = \
                    self.correlators.get_vertex_green_convolution(
                        green=self.green_aux.inverse(), component=con_comp,
                        spin=spins, position_freq=(1, 1),
                        vertex=self.three_point_vertex)
                # XXX: function passed should be different from susceptibility
                self.three_point_vertex = \
                    self.correlators.get_vertex_green_convolution(
                        green=self.susceptibility_aux.inverse(), component=con_comp,
                        spin=spins, position_freq=(1, 2),
                        vertex=self.three_point_vertex)

    def get_mirrored_three_point_vertex(self, component: tuple, spin: tuple) -> None:
        # Here we simply implement the symmetry relations written in the
        # auxiliary dual boson paper of Feng
        conj_component = (1 - component[0], 1 - component[1], 1 - component[2])
        return self.three_point_vertex[spin][conj_component].T.conj()

    def compute_polarization_dual(self, green: "fg.FrequencyGreen" = None
                                  ) -> None:
        if green is None:
            green = self.green_dual
        # TODO: Polarization
        #      [ ] 0. write the fourier transform of the convolution.
        #      [ ] 1. implement convolution of vertex with vertices and green's
        #             functions

    def compute_sigma_dual(self, green: "fg.FrequencyGreen" = None) -> None:
        if green is None:
            green = self.green_dual

        # TODO: sigma
        #      [ ] 0. write the foutier transform of the convolutions.
        #      [ ] 1. implement convolution of vertex with vertices and green's
        #             functions

    def solve(self, err_tol=1e-7, iter_max=100):
        self.__solve__(err_tol=err_tol, iter_max=iter_max)

    def save(self, fname: str, dir_: str, dataname: str,
             save_input_param: bool = True, save_aux_data: bool = False
             ) -> None:
        self.__save__(fname=fname, dir_=dir_, dataname=dataname,
                      save_input_param=save_input_param,
                      save_aux_data=save_aux_data)

        freq = {'freq_min': self.hyb_aux.freq[0],
                'freq_max': self.hyb_aux.freq[-1],
                'N_freq': len(self.hyb_aux.freq)}

        time = {'time_min': self.time[0],
                'time_max': self.time[-1],
                'N_time': len(self.time)}
        hd5.add_attrs(fname, "/", {'freq': freq, 'time': time})

        hd5.add_dict_data(fname, f"{dir_}/{dataname}", 'three_point_vertex',
                          self.three_point_vertex)
        hd5.add_dict_data(fname, f"{dir_}/{dataname}", 'four_point_vertex',
                          self.four_point_vertex)

    def load(self, fname: str, dir_: str, dataname: str,
             load_input_param: bool = True, load_aux_data: bool = False
             ) -> None:
        print('bla')
        self.__load__(fname=fname, dir_=dir_, dataname=dataname,
                      load_input_param=load_input_param,
                      load_aux_data=load_aux_data)

        if load_input_param:

            grid = hd5.read_attrs(fname, '/')
            try:
                self.time = np.linspace(
                    grid['time']['time_min'], grid['time']['time_max'],
                    grid['time']['N_time'])
            except KeyError:
                print(KeyError)
        temp = hd5.read_dict_data(
            fname, f"{dir_}/{dataname}", 'three_point_vertex')
        for key in self.three_point_vertex:
            self.three_point_vertex[key] = temp[f"{key}"]

        temp = hd5.read_dict_data(
            fname, f"{dir_}/{dataname}", 'four_point_vertex')
        for key in self.four_point_vertex.keys():
            self.four_point_vertex[key] = temp[f"{key}"]


# %%
if __name__ == '__main__':
    time_param = {'time_min': -10, 'time_max': 10, 'N_time': 1001}
    auxTrilex = AuxiliaryDualTRILEX(
        filename='ForAuxTrilex.h5', dir_='', dataname='trilex',
        time_param=time_param)
# %%
if __name__ == '__main__':
    import src.util.figure as fu
    import colorcet as cc
    plot = fu.FigureTheme(cmap=cc.cm.bgy)

    plot.create_single_panel(xlabel='t\'',
                             ylabel='t',
                             zlabel='$G^{R}$')
    plot.ax.imshow(auxTrilex.three_point_vertex[
        ('up', 'up', 'ch')][:, :, 0, 0, 0].imag, cmap=cc.cm.bgy)
    plt.show()

# %%
