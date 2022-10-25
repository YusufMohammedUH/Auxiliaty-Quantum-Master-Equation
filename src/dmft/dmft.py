from abc import abstractmethod
from typing import Dict, Union, Callable, Tuple
import numpy as np
from scipy.integrate import simps
import src.greens_function.frequency_greens_function as fg
import src.greens_function.dos_util as du
import src.auxiliary_mapping.optimization_auxiliary_hybridization as opt
import src.util.hdf5_util as hd5
import src.greens_function.convert_keldysh_components as conv


class DMFTBase:
    """Class for the DMFT solver using the auxiliary master equation method.


    Parameters
    ----------
    parameters : Dict
        Contains all the parameters needed for the calculation.

    hyb_leads : Union[fg.FrequencyGreen, None], optional
        Lead hybridization function, by default None

    Attributes
    ----------
    parameters : Dict
        Contains all the parameters needed for the calculation.

    hyb_leads : Union[fg.FrequencyGreen, None]
        Lead hybridization function.

    hyb_dmft : fg.FrequencyGreen
        DMFT hybridization function.

    hyb_sys : fg.FrequencyGreen
        Total system hybridization(hyb_dmft+hyb_leads) function.

    green_sys : fg.FrequencyGreen
        Green's function of the system.

    green_aux : fg.FrequencyGreen
        Green's function of the auxiliary system.

    self_energy_int : fg.FrequencyGreen
        Self-energy of the system.

    T_mat : np.ndarray
        Hopping matrix of the auxiliary system.

    U_mat : np.ndarray
        Onsite interaction matrix of the auxiliary system.

    err_iterations : List
        List of the errors at each iteration.

    Raises
    ------
    ValueError
        Only single orbital included for now. Therefor must be
        nsite = 2*Nb + 1.
    """

    def __init__(self, parameters: Dict,
                 hyb_leads: Union[fg.FrequencyGreen, None] = None) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        self.parameters = parameters
        self.err_iterations = []
        self.n = None
        if hyb_leads is None:
            freq = np.linspace(parameters['freq']['freq_min'],
                               parameters['freq']['freq_max'],
                               parameters['freq']['N_freq'])
            freq.flags.writeable = False
            self.hyb_leads = fg.FrequencyGreen(freq)
            self.hyb_dmft = fg.FrequencyGreen(freq)

            self.green_sys = fg.FrequencyGreen(freq)
            self.self_energy_int = fg.FrequencyGreen(freq)
        else:
            self.hyb_leads = hyb_leads
            self.hyb_dmft = fg.FrequencyGreen(self.hyb_leads.freq)

            self.green_sys = fg.FrequencyGreen(self.hyb_leads.freq)
            self.self_energy_int = fg.FrequencyGreen(self.hyb_leads.freq)

    def get_bath(self, param: Union[Dict, None] = None,
                 bath_hyb_function: Callable[
        [float, float, float, float, float],
            Tuple[np.ndarray, np.ndarray]] = du.flat_bath_retarded
    ) -> fg.FrequencyGreen:
        """Return a hybridization function for the passed arguments.

        Parameters
        ----------
        param : Union[Dict, None], optional
            Dictionary containing the parameters passed to the function
            bath_hyb_function, by default None.
            If None than the parameters specified in self.parameters['leads']
            are used.

        bath_hyb_function : Function, optional
            Function returning the keldysh and retarded components
            (tuple of two np.ndarray's of complex numbers), by default
            du.flat_bath_retarded

        Returns
        -------
        out:    fg.FrequencyGreen
            Hybridization of the system leads.
        """
        if param is None:
            param = self.parameters
        tmp = fg.FrequencyGreen(self.green_sys.freq)
        for mu in param['leads']['mu']:
            args = np.array([param['leads']['e0'], mu,
                            param['leads']['beta'], param['leads']['D'],
                            param['leads']['gamma']],
                            dtype=np.float64)
            tmp += du.set_hybridization(self.hyb_leads.freq, bath_hyb_function,
                                        args)
        return tmp

    def __solve__(self, solver_parameters=()) -> None:
        """Solve the auxiliary dynamic mean-field theory for a given set of
        parameters and a supplied object of class Correlators.

        Parameters
        ----------
        correlators : corr.Correlators
            Object of class Correlators containing routines to calculate the
            single particle greens function and the Lindblad operator.

        U_mat : Union[np.ndarray, None], optional
            Interaction matrix, by default None
        """

        self.err_iterations = []
# ------------------------- initial Green's function ------------------------ #
        args = np.array([self.parameters['leads']['e0'],
                         0,
                         self.parameters['leads']['beta'], 1.0, 1.0],
                        dtype=np.float64)
        green_tmp = du.set_hybridization(
            self.green_sys.freq, du.lorenzian_bath_retarded, args)
        self.green_sys.dyson(self_energy=green_tmp)
        for ii in range(self.parameters['selfconsistency']['max_iter']):

            # -------------- Update hybridization --------------------------- #
            self.hyb_dmft = fg.FrequencyGreen(
                self.hyb_leads.freq, green_tmp.retarded *
                (self.parameters['system']['v']**2),
                green_tmp.keldysh * (self.parameters['system']['v']**2))
# -------------------------- Calculate occupation --------------------------- #
            green_lesser = conv.get_lesser_from_keldysh(self.green_sys)
            self.n = simps(green_lesser.imag, self.green_sys.freq)
# -------------------- New Green's function from solver --------------------- #
            solver_parameters = self.impurity_solver(*solver_parameters)

# ------------------------- Update Geen's function -------------------------- #
            green_tmp = fg.FrequencyGreen(self.green_sys.freq, retarded=(
                (1.0 - self.parameters['selfconsistency']['mixing'])
                * self.green_sys.retarded
                + self.parameters['selfconsistency']['mixing']
                * green_tmp.retarded),
                keldysh=((1.0 - self.parameters['selfconsistency']['mixing'])
                         * self.green_sys.keldysh
                         + self.parameters['selfconsistency']['mixing']
                         * green_tmp.keldysh))
# ------------------------- print values of interest ------------------------ #
            spectral_weight = round((-1 / np.pi) * simps(
                self.green_sys.retarded.imag, self.green_sys.freq), 8)

            self.err_iterations.append(opt.cost_function(self.green_sys,
                                                         green_tmp,
                                                         normalize=False))
            err_iter_print = round(self.err_iterations[-1], int(
                5 - np.log10(self.parameters['selfconsistency']['err_tol'])))
            print(
                f"Iteration No.: {ii}   |    Error: "
                + "{:.7}   |   ".format(err_iter_print)
                + f"Spectral weight: {spectral_weight}  |   "
                + f"occupation:  {self.n}")
# ----------------------- check convergency condition ----------------------- #
            if self.err_iterations[-1]  \
                    < self.parameters['selfconsistency']['err_tol']:
                self.green_sys = green_tmp
                break

# ---------------- update dmft hybridization to last iteration -------------- #
        self.hyb_dmft = fg.FrequencyGreen(
            self.hyb_leads.freq, self.green_sys.retarded *
            (self.parameters['system']['v']**2),
            self.green_sys.keldysh * (self.parameters['system']['v']**2))

    @abstractmethod
    def impurity_solver(self, args: Tuple) -> None:
        pass

    def save(self, fname: str) -> None:
        """Save data to file

        Parameters
        ----------
        fname : str
            Name of HDF5 file.
        """
        hd5.create_hdf5(fname)
        hd5.add_attrs(fname, '/', self.parameters['freq'])
        hd5.add_data(fname, '/', 'convergence', self.err_iterations)
        hd5.add_attrs(fname, '/convergence',
                      self.parameters['selfconsistency'])

        self.green_sys.save(fname, '/system', 'green_sys', savefreq=False)
        self.hyb_dmft.save(fname, '/system', 'hyb_dmft', savefreq=False)
        hd5.add_attrs(fname, '/system', self.parameters['system'])

        if 'leads' in self.parameters:
            self.hyb_leads.save(fname, '/system', 'hyb_leads', savefreq=False)
            hd5.add_attrs(fname, '/system', self.parameters['leads'])

        self.save_child_data(fname)

    def load(self, fname: str, read_parameters: bool = False) -> None:
        """Load data from file

        Parameters
        ----------
        fname : str
            Name of HDF5 file.
        """
        self.err_iterations = hd5.read_data(fname, '/', 'convergence')

        self.green_sys.load(fname, '/system', 'green_sys', readfreq=False)
        self.hyb_dmft.load(fname, '/system', 'hyb_dmft', readfreq=False)
        if 'leads' in self.parameters:
            self.hyb_leads.load(fname, '/system', 'hyb_leads', readfreq=False)

        if read_parameters:
            self.parameters = {}
            self.parameters['selfconsistency'] = hd5.read_attrs(
                fname, '/convergence')
            self.parameters['system'] = hd5.read_attrs(fname, '/system')
            self.parameters['freq'] = hd5.read_attrs(fname, '/')

            if 'leads' in self.parameters:
                self.parameters['leads'] = hd5.read_attrs(fname, '/system')

    @abstractmethod
    def save_child_data(self, fname: str) -> None:
        pass

    @abstractmethod
    def load_child_data(self, fname: str, read_parameters: bool = False
                        ) -> None:
        pass


class DMFT_GW(DMFTBase):
    """_summary_

    _extended_summary_
    """

    def __init__(self, parameters: Dict) -> None:
        super().__init__(parameters)