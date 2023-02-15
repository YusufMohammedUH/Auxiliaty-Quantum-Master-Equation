# %%
from typing import Dict, Optional
import matplotlib.pyplot as plt
import src.greens_function.frequency_greens_function as fg
import src.greens_function.correlation_functions as corr
import src.util.hdf5_util as hd5

import src.impurity_solver.auxiliary_solver_base as aux_base
import src.dmft.auxiliary_dmft as aux_dmft

# Dual TRILEX solver:
#  [X] 1. inherit from AuxiliaryDualSolverBase:
#       [X] 0. get g_aux, hyb_sys, hyb_aux
#       [X] 1. calculate auxiliary susceptibility
#       [X] 2. calculate auxiliary polarization
#       [X] 3. calculate non-interacting dual Green's function and bosonic
#         propagator
#
#       [X] 7. calculate the dressed dual bosonic propagator
#       [X] 9. calculate the dressed dual Green's function
#   (optional) update self-consistently 6. with dressed dual Green's function
#              until convergence
#       [X] 10. calculate the system Green's function
#  [X] 2. calculate three point vertex
#  [X] 2. calculate dual polarization
#  [X] 3. calculate dual self-energy


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

    three_point_vertex : fg.FrequencyGreen
        Dual four-point vertex

    four_point_vertex : fg.FrequencyGreen
        Dual four-point vertex

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
                 U_trilex: Optional[Dict] = None,
                 green_aux: Optional[fg.FrequencyGreen] = None,
                 hyb_sys: Optional[fg.FrequencyGreen] = None,
                 hyb_aux: Optional[fg.FrequencyGreen] = None,
                 correlators: Optional[corr.Correlators] = None,
                 keldysh_comp: str = 'keldysh') -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        super().__init__(filename=filename, dir_=dir_, dataname=dataname,
                         load_input_param=load_input_param,
                         load_aux_data=load_aux_data,
                         U_trilex=U_trilex,
                         green_aux=green_aux,
                         hyb_sys=hyb_sys,
                         hyb_aux=hyb_aux,
                         correlators=correlators,
                         keldysh_comp=keldysh_comp)

        if (filename is None) and (dir_ is None) and (dataname is None):
            self.three_point_vertex = {('up', 'up', 'ch'): None,
                                       ('up', 'do', 'x'): None,
                                       ('up', 'do', 'y'): None,
                                       ('up', 'up', 'z'): None}
            self.four_point_vertex = {}

    def compute_three_point_vertex(self) -> None:
        """Calculate the three point vertex
        """
        for spins in self.three_point_vertex:
            self.three_point_vertex = self.correlators.get_three_point_vertex(
                self.green_aux.freq, spins, return_=True)

    def compute_polarization_dual(self, green: "fg.FrequencyGreen" = None
                                  ) -> None:
        if green is None:
            green = self.green_dual
        # XXX: Here the polarization is calculated from the dual Green's

        # if self.keldysh_comp == 'lesser':
        #     self.polarization_dual = polarizability_tmp
        # elif self.keldysh_comp == 'keldysh':
        #     self.polarization_dual.retarded = polarizability_tmp.retarded
        #     self.polarization_dual.keldysh = polarizability_tmp.get_keldysh()

    def compute_sigma_dual(self, green: "fg.FrequencyGreen" = None) -> None:
        if green is None:
            green = self.green_dual
        # XXX: Here the polarization is calculated from the dual Green's

        # if self.keldysh_comp == "lesser":
        #     self.sigma_dual = sigma_dual_tmp
        # elif self.keldysh_comp == "keldysh":
        #     self.sigma_dual.retarded = sigma_dual_tmp.retarded
        #     self.sigma_dual.keldysh = sigma_dual_tmp.get_keldysh()
        # plt.plot(self.sigma_dual.retarded.imag)

    def solve(self, err_tol=1e-7, iter_max=100):
        self.__solve__(err_tol=err_tol, iter_max=iter_max)

    def save(self, fname: str, dir_: str, dataname: str,
             save_input_param: bool = True, save_aux_data: bool = False
             ) -> None:

        super().save(fname=fname, dir_=dir_, dataname=dataname,
                     save_input_param=save_input_param,
                     save_aux_data=save_aux_data)

        hd5.add_dict_data(fname, f"{dir_}/{dataname}", 'three_point_vertex',
                          self.three_point_vertex)
        hd5.add_dict_data(fname, f"{dir_}/{dataname}", 'four_point_vertex',
                          self.four_point_vertex)

    def load(self, fname: str, dir_: str, dataname: str,
             load_input_param: bool = True, load_aux_data: bool = False
             ) -> None:

        super().load(fname=fname, dir_=dir_, dataname=dataname,
                     load_input_param=load_input_param,
                     load_aux_data=load_aux_data)

        temp = hd5.read_dict_data(
            fname, f"{dir_}/{dataname}", 'three_point_vertex')
        for key in self.three_point_vertex:
            self.three_point_vertex[key] = temp[f"{key}"]

        temp = hd5.read_dict_data(
            fname, f"{dir_}/{dataname}", 'four_point_vertex')
        for key in self.four_point_vertex.keys():
            self.four_point_vertex[key] = temp[f"{key}"]


if __name__ == '__main__':
    auxTrilex = AuxiliaryDualTRILEX(
        filename='ForAuxTrilex.h5', dir_='', dataname='trilex')
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
