import pytest
import numpy as np
from src import frequency_greens_function as fg


def test_FrequencySystem_freq_raise_TypeError():
    ws = [i for i in range(10)]
    with pytest.raises(TypeError):
        fg.FrequencyGreen(ws)


def test_FrequencySystem_retarded_raise_ValueError():
    ws = [i for i in range(10)]
    with pytest.raises(ValueError):
        fg.FrequencyGreen(np.array(ws), np.array(ws[2:]))


def test_FrequencySystem_retarded_raise_ValueError():
    ws = [i for i in range(10)]
    with pytest.raises(ValueError):
        fg.FrequencyGreen(np.array(ws), np.array(ws[2:]))

# TODO: 1. write test for dyson function
#       2.       -"-      set_green_from_auxiliary
#       3.       -"-      get_self_enerqy


if __name__ == "__main__":

    pytest.main("-v test_frequency_greens_function.py")
