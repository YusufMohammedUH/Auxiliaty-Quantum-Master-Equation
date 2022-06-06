from typing import Set, List
import itertools
import numpy as np
from numba import njit
from sympy import Matrix


@njit(cache=True)
def get_ranges(n_tmax, list_of_times):
    ranges = np.zeros((len(list_of_times), 2), dtype=np.int64)
    if len(list_of_times) == 1:
        ranges[0] = [n_tmax * list_of_times[0], n_tmax * list_of_times[0]]
        return ranges

    for i, li in enumerate(list_of_times):
        if li == 0:
            ranges[i] = [-n_tmax, n_tmax]
        elif li == -1:
            ranges[i] = [-n_tmax, 0]
        elif li == 1:
            ranges[i] = [n_tmax, n_tmax]
    return ranges


@njit(cache=True)
def get_greens_component_times(real_times, time_ranges, n_converted):
    times_value = np.zeros(len(n_converted), dtype=np.float64)
    indices = np.zeros(len(n_converted), dtype=np.int64)
    for i in range(len(n_converted)):
        time_range = time_ranges[i][1] - time_ranges[i][0]
        if time_range == 0:
            j = time_ranges[i][0]
            times_value[i] = np.sign(j) * real_times[np.abs(j)]
            indices[i] = len(real_times) - 1 + j
        else:
            j = time_ranges[i][0] + n_converted[i]
            times_value[i] = np.sign(j) * real_times[np.abs(j)]
            indices[i] = len(real_times) - 1 + j
    return times_value, indices


def get_permutation(li):
    return list(set(map(lambda x: tuple(x), itertools.permutations(
        li, r=len(li)))))


def make_tuples(depth, shape, start=0):
    if depth == 0:
        yield ()
    else:
        for x in range(0, shape[0]):
            for t in make_tuples(depth - 1, shape[1:], x + 1):
                yield (x,) + t


def get_permutations_sign(n):
    index = list(np.ndindex(n, n))
    index_string_array = np.array(
        list(map(lambda x: 'a' + "".join(x),
                 list(map(lambda y: (str(y[0]), str(y[1])), index)))))
    index_string_array = index_string_array.reshape((n, n))
    A = Matrix(index_string_array)
    A_determinante_str = A.det().__str__()

    permutation = {}
    topermute = A_determinante_str.replace("-", "+").split(" + ")
    for i, s in enumerate(topermute):
        s_tuple = tuple(s.replace('a', "").split('*'))
        sign = ""
        if i == 0:
            sign = 1
        else:
            if A_determinante_str.split(s)[0][-2] == '-':
                sign = -1
            else:
                sign = 1
        key = [None] * len(s_tuple)

        for j in s_tuple:
            key[int(j[1])] = int(j[0])

        permutation[tuple(key)] = sign
    return permutation


def contour_ordering(list_of_parameters):
    return sorted(list_of_parameters, key=(
        lambda x: (-x[0], x[1]) if x[0] == 1 else (-x[0], -x[1])))


def steady_state_contour(contour_parameters):
    contour_parameters_new = list(
        map(lambda x: (x[0], x[1] - contour_parameters[-1][1], *x[2:]),
            contour_parameters))
    real_times = list(map(lambda x: x[1], contour_parameters_new))[:-1]
    return contour_parameters_new, real_times


def quantum_regresion_ordering(list_of_parameters, times):

    idx = None
    for i, x in enumerate(list_of_parameters):
        if x[1] < 0:
            idx = i

    if idx is None:
        return list_of_parameters, times
    else:
        return (list_of_parameters[(idx + 1):]
                + list_of_parameters[:(idx + 1)],
                times[(idx + 1):] + times[:(idx + 1)])


def get_times_direction(li):
    tmp = [1 if i != len(li) - 1 else 0 for i in range(len(li))]
    times = []
    if li[-1] == 0:
        times.append(tmp.copy())
    else:
        times.append([-1 * i for i in tmp])

    if (1 in li) and (0 in li):
        for i in range(li.count(1)):
            tmp[i] = -1 * tmp[i]
            times.append(tmp.copy())

    return times


def get_time_parametrization(li):
    n = len(li)
    t_normal_order_list = [x for x in zip(range(n - 1), range(1, n))]
    time_pair_normal_order = {(x[0], None): x for x in t_normal_order_list}
    time_pair_normal_order_sum = {(i[0], j[0]): (i[0], j[1]) for i, j in zip(
        t_normal_order_list[:-1], t_normal_order_list[1:])}
    time_pair_component_list = [(i, j) for i, j in zip(li[:-1], li[1:])]
    time_pair_component = {}
    for i, x in enumerate(time_pair_component_list):
        time_pair_component[i] = {}
        for t_key in time_pair_normal_order:
            if x == time_pair_normal_order[t_key]:
                time_pair_component[i][t_key] = 1
            elif x[::-1] == time_pair_normal_order[t_key]:
                time_pair_component[i][t_key] = -1
        for t_key in time_pair_normal_order_sum:
            if x == time_pair_normal_order_sum[t_key]:
                time_pair_component[i][t_key] = 1
            elif x[::-1] == time_pair_normal_order_sum[t_key]:
                time_pair_component[i][t_key] = -1

    return time_pair_component


def choose_components(components: List, contour_symmetries: dict) -> None:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    components : List
        _description_
    contour_symmetries : dict
        _description_
    """
    for key in contour_symmetries:
        value = contour_symmetries[key]
        if key in components:
            if value is not None:
                contour_symmetries[value] = key
                contour_symmetries[key] = None


def get_branch_combinations(n: int, contour_ordered: bool = False) -> List:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    n : int
        _description_
    contour_ordered : bool, optional
        _description_, by default False

    Returns
    -------
    List
        _description_
    """
    # forward branch 0/ backward branch 1
    # Returns all possible branch combinations
    combos = []
    for i in range(2**n):
        compination = list(map(int, bin(i).replace("0b", "")))
        padding = [0 for i in range(n - len(compination))]
        combos.append(padding + compination)
    combos = np.array(combos)
    if contour_ordered:
        mask = list(map(lambda x: np.all(
            x == sorted(x, reverse=True)), combos))
        return list(map(tuple, combos[mask]))
    return list(map(tuple, combos))


def t_max_possitions(n: int) -> Set:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    n : int
        _description_

    Returns
    -------
    Set
        _description_
    """
    t_max = list(map(tuple, itertools.permutations(
        [0 if i != n - 1 else 1 for i in range(n)], r=n)))
    return set(t_max)


def find_index_max_time(list_of_parameters: List) -> int:
    """_summary_

    _extended_summary_

    Parameters
    ----------
    list_of_parameters : List
        _description_

    Returns
    -------
    int
        _description_
    """
    return list_of_parameters.index(
        max(list_of_parameters, key=(lambda x: x[1])))
