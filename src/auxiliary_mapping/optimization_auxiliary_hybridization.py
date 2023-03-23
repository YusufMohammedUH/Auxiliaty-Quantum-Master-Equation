# %%
from typing import Union, Tuple, List, Dict
import numpy as np
import src.auxiliary_mapping.auxiliary_system_parameter as auxp
import src.greens_function.frequency_greens_function as fg
import src.greens_function.dos_util as du
from scipy.integrate import simps
from scipy.optimize import minimize, Bounds
from scipy.optimize.optimize import OptimizeResult
import matplotlib.pyplot as plt
# TODO: 1. enable complex optimization
# TODO: 2. use a optimization which converges reliably for NB>2
#          -> change to more reliable minimization scheme!
#              e.g. Stochastic minimization
# XXX: optimization doesn't converge reliably for Nb>2


def cost_function(hybridization: fg.FrequencyGreen,
                  auxiliary_hybridization: fg.FrequencyGreen,
                  weight: Union[np.ndarray, None] = None,
                  normalize: bool = True) -> float:
    """Cost function returns the weoght, integrated, elementwise squared
    difference between the retarded and Keldysh component of two supplied
    frequency, single particle Green's function type objects.

    Parameters
    ----------
    hybridization : frequency_greens_function.FrequencyGreen
        First frequency, single particle Green's function type object

    auxiliary_hybridization : frequency_greens_function.FrequencyGreen
        Second frequency, single particle Green's function type object

    weight : numpy.ndarray, optional
        Weights of importance, e.g. one could focus on values close to the
        fermi edge, by default None

    Returns
    -------
    out: float
        weight, renormalized difference between between two
        frequency_greens_function.FrequencyGreen objects
    """
    if weight is None:
        weight = np.ones(hybridization.freq.shape)
    norm = 1.0
    if normalize:
        norm = 1.0 / (simps(np.square(hybridization.retarded.imag)
                            + np.square(hybridization.keldysh.imag),
                            hybridization.freq))

    diff_ret = (np.square((hybridization.retarded -
                           auxiliary_hybridization.retarded).imag))
    diff_ret = diff_ret[~np.isnan(diff_ret)]

    diff_kel = (np.square((hybridization.keldysh -
                           auxiliary_hybridization.keldysh).imag))
    diff_kel = diff_kel[~np.isnan(diff_kel)]

    return norm * simps((diff_ret + diff_kel) * weight, hybridization.freq)


def optimize_subroutine(x0: np.ndarray, *args) -> float:
    """Optimization function of the hybridization function.
    Set the current hybridization function from x0 and compare to the target
    hybridization function, with parameters supplied in args.
    The difference between target and current hybridization is returned.

    Parameters
    ----------
    x0 : array_like
        Contains the parameters to be optimized in order to obtain the
        optimal approximation to a hybridization function

    args: array_like
        Contains arguments necessary to set the problem to optimize over.

    Returns
    -------
    out: float
        Value of discrepancy between current and target hybridization.
    """
    Nb = args[0]
    hybridization = args[1]
    weight = args[2]

    es = np.array(x0[0:(Nb)])
    ts = np.array(x0[(Nb):(2 * Nb)])
    gammas = np.array(x0[(2 * Nb):])

    aux = auxp.AuxiliarySystem(Nb, hybridization.freq)
    aux.set_ph_symmetric_aux(es, ts, gammas)

    green = fg.FrequencyGreen(hybridization.freq, keldysh_comp="keldysh")
    green.set_green_from_auxiliary(aux)

    hyb_aux = green.get_self_enerqy()

    return cost_function(hybridization, hyb_aux, weight)


def optimization_ph_symmertry(Nb: int, hybridization: fg.FrequencyGreen,
                              weight: Union[np.ndarray, None] = None,
                              x_start: Union[np.ndarray, None] = None,
                              N_try: int = 1, dtype: type = float,
                              bounds: Union[List[Tuple],
                                            Bounds, None] = None,
                              constraints: Union[Tuple, None] = None,
                              options: Union[Dict, None] = None
                              ) -> OptimizeResult:
    """Approximation of the supplied hybridization function by a auxiliary
    hybridization function of an auxiliary system with Nb left and right
    auxiliary sites.

    This routine works currently only reliably for Nb=1

    Parameters
    ----------
    Nb : int
        Sets number of auxiliary sites (2*Nb)

    hybridization : frequency_greens_function.FrequencyGreen
        Hybridization function to be approximated

    weight : numpy.ndarray, optional
        Weight for calculating the cost function, by default None

    x_start : list, optional
        initial guess parameters of the auxiliary system, by default None

    N_try : int, optional
        Number of tries for a converging approximation, by default 1

    dtype : (float,complex), optional
        data type of auxiliary system parameters, by default float

    bounds : list of tuples, optional
        bounds for the optimization, by default None

    constraints : tuple, optional
        constraints for the optimization, by default None

    options : dict, optional
        options for the optimization, by default None

    Returns
    -------
    result: scipy.optimize.optimize.OptimizeResult
        Result of the optimization of a given hybridization function
        by an auxiliary hybridization function
        result.x contains the converged parameres ouf the auxiliary system

    Raises
    ------
    ValueError
        If the approximation didn't converge an error is raised
    """
    if constraints is None:
        constraints = ()
    if options is None:
        options = {"disp": True, "maxiter": 200,
                   "return_all": True,
                   'gtol': 1e-5}
    aux_tmp = auxp.AuxiliarySystem(Nb, hybridization.freq)
    if x_start is None or N_try == 1:
        es = np.ones(Nb, dtype=dtype)
        ts = np.ones(Nb, dtype=dtype)
        gammas = np.ones(aux_tmp.N_gamma, dtype=dtype)
        x_start = np.array([*es, *ts, *gammas])
    if weight is None:
        weight = np.ones(hybridization.freq.shape)
    args = (Nb, hybridization, weight)
    result = minimize(optimize_subroutine, x_start, bounds=bounds,
                      constraints=constraints, method='SLSQP', args=args,
                      options=options,
                      )

    nn = 1
    while (N_try > nn) and (~result.success):
        result = minimize(optimize_subroutine, x_start, args=args)
        nn += 1
        print("No. of tries: ", nn, ", converged: ", result.success)
    if not result.success:
        raise ValueError(
            "Minimization of auxiliary hybridization not converged!")
    return result


def get_aux_hyb(res: np.ndarray, Nb: int, freq: np.ndarray
                ) -> fg.FrequencyGreen:
    """Returns the hybridization function of an auxiliary system


    Parameters
    ----------
    res : numpy.ndarray
        contains the parameters for determening E,Gamma1 and Gamma2 of the
        auxiliary system
    Nb : int
        Determens number of auxiliary sites (2*Nb)
    freq : numpy.ndarray
        Frequency grid

    Returns
    -------
    frequency_greens_function.FrequencyGreen
        Hybridization function of the auxiliary system for supplied parameters
    """
    es = np.array(res[0:(Nb)])
    ts = np.array(res[(Nb):(2 * Nb)])
    gammas = np.array(res[(2 * Nb):])
    aux = auxp.AuxiliarySystem(Nb, freq)
    aux.set_ph_symmetric_aux(es, ts, gammas)

    green = fg.FrequencyGreen(freq, keldysh_comp="keldysh")
    green.set_green_from_auxiliary(aux)

    return green.get_self_enerqy()


def get_aux(res: np.ndarray, Nb: int, freq: np.ndarray
            ) -> auxp.AuxiliarySystem:
    """Returns a auxiliary system

    Parameters
    ----------
    res : numpy.ndarray
        contains the parameters for determening E,Gamma1 and Gamma2 of the
        auxiliary system
    Nb : int
        Determens number of auxiliary sites (2*Nb)
    freq : numpy.ndarray
        Frequency grid

    Returns
    -------
    auxiliary_system_parameter.AuxiliarySystem
        Auxiliary system object for supplied parameters
    """
    es = np.array(res[0:(Nb)])
    ts = np.array(res[(Nb):(2 * Nb)])
    gammas = np.array(res[(2 * Nb):])
    aux = auxp.AuxiliarySystem(Nb, freq)
    aux.set_ph_symmetric_aux(es, ts, gammas)

    return aux


class AuxiliaryHybridization():
    # XXX: This class only works for single orbital and site impurities
    def __init__(self, Nb: int, x_start: Union[None, np.ndarray] = None,
                 U_mat: Union[np.ndarray, None] = None,
                 half_filling: bool = True) -> None:
        self.Nb = Nb
        self.x_start = x_start
        self.aux_sys = None
        if U_mat.shape == (2 * Nb + 1, ):
            self.U_diag = U_mat
        elif U_mat.shape == (2 * Nb + 1, 2 * Nb + 1):
            self.U_diag = np.diag(U_mat)
        elif U_mat.shape == (2 * Nb + 1, 2 * Nb + 1, 2 * Nb + 1, 2 * Nb + 1):
            self.U_diag = np.diag(np.einsum('ijji->ij', U_mat))
        else:
            raise ValueError("U_mat has wrong shape")
        self.half_filling = half_filling

    def update(self, hyb: fg.FrequencyGreen,
               options: Union[None, Dict] = None
               ) -> fg.FrequencyGreen:
        if 2 * self.Nb + 1 != self.U_diag.shape[0]:
            raise ValueError("U_diag and E have wrong shape")
        if hyb.keldysh_comp == 'keldysh':
            hyb_tmp = hyb
        else:
            hyb_tmp = fg.FrequencyGreen(freq=hyb.freq,
                                        retarded=hyb.retarded,
                                        keldysh=hyb.get_keldysh(),
                                        keldysh_comp="keldysh")
        optimal_param = optimization_ph_symmertry(Nb=self.Nb,
                                                  hybridization=hyb_tmp,
                                                  x_start=self.x_start,
                                                  options=options)
        self.x_start = np.copy(optimal_param.x)
        self.aux_sys = get_aux(self.x_start, self.Nb, hyb.freq)
        hyb_aux = fg.get_hyb_from_aux(self.aux_sys, hyb.keldysh_comp)

        if self.half_filling:
            self.aux_sys.E -= np.diag(self.U_diag / 1.0)

        return hyb_aux


if __name__ == "__main__":
    # Setting target hybridization
    e0 = 0
    mu = 0.5
    beta = 100
    N_freq = 1001
    freq_max = 10
    D = 3
    gamma = 1
    freq = np.linspace(-freq_max, freq_max, N_freq)

    flat_hybridization_retarded1 = np.array(
        [du.flat_bath_retarded(w, e0, D, gamma) for w in freq])
    flat_hybridization_keldysh1 = np.array(
        [1.j * (1. / np.pi) * (1. - 2. * du.fermi(w, e0, mu, beta)) *
         np.imag(du.flat_bath_retarded(w, e0, D, gamma)) for w in freq])

    flat_hybridization_retarded2 = np.array(
        [du.flat_bath_retarded(w, e0, D, gamma) for w in freq])
    flat_hybridization_keldysh2 = np.array(
        [1.j * (1. / np.pi) * (1. - 2. * du.fermi(w, e0, -mu, beta)) *
         np.imag(du.flat_bath_retarded(w, e0, D, gamma)) for w in freq])

    hybridization = fg.FrequencyGreen(
        freq, flat_hybridization_retarded1 + flat_hybridization_retarded2,
        flat_hybridization_keldysh1 + flat_hybridization_keldysh2)
    options = {"disp": True, "maxiter": 500, 'ftol': 1e-5}

    # Calculating auxiliary hyridization for Nb =1
    try:
        Nb = 1
        result_nb1 = optimization_ph_symmertry(
            Nb, hybridization, options=options)
        aux_nb1 = get_aux(result_nb1.x, Nb, freq)
        hyb_aux_nb1 = fg.get_hyb_from_aux(aux_nb1, keldysh_comp='keldysh')
    except ValueError:
        print(f"Minimization for Nb = {Nb}, not converged.")

    # Calculating auxiliary hyridization for Nb =2
    try:
        Nb = 2
        result_nb2 = optimization_ph_symmertry(
            Nb, hybridization, options=options)
        aux_nb2 = get_aux(result_nb2.x, Nb, freq)
        hyb_aux_nb2 = fg.get_hyb_from_aux(aux_nb2, keldysh_comp='keldysh')
    except ValueError:
        print(f"Minimization for Nb = {Nb}, not converged.")

    # # Calculating auxiliary hyridization for Nb =3
    # try:
    #     Nb = 3
    #     result_nb3 = optimization_ph_symmertry(
    #         Nb, hybridization, options=options)
    #     aux_nb3 = get_aux(result_nb3.x, Nb, freq)
    #     hyb_aux_nb3 = fg.get_hyb_from_aux(aux_nb3)
    # except ValueError:
    #     print(f"Minimization for Nb = {Nb}, not converged.")

    plt.figure()
    plt.plot(hybridization.freq, hybridization.get_spectral_func())
    plt.plot(hybridization.freq, hyb_aux_nb1.get_spectral_func())
    plt.plot(hybridization.freq, hyb_aux_nb2.get_spectral_func())
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$-\frac{1}{\pi}Im\Delta^R(\omega)$")
    plt.legend([r"$exact$", r"$Nb=2$", r"$Nb=4$"])
    plt.show()

    plt.figure()
    plt.plot(hybridization.freq, hybridization.keldysh.imag)
    plt.plot(hybridization.freq, hyb_aux_nb1.keldysh.imag)
    plt.plot(hybridization.freq, hyb_aux_nb2.keldysh.imag)
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$Im\Delta^K(\omega)$")
    plt.legend([r"$exact$", r"$Nb=2$", r"$Nb=4$"])

# %%
