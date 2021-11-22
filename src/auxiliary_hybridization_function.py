# ##############################################################################
#
# Auxiliaty Quantum Master Equation
#
# Copyright (c) 2021 Yusuf Mohammed
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################
from scipy.sparse import diags
import numpy as np


class AuxiliarySystem:

    # def __init__(self, Nb, ws, E, Gamma1, Gamma2) -> None:
    def __init__(self, Nb) -> None:
        self.Nb = Nb
        # self.ws = ws
        # self.E = E
        # self.Gamma1 = Gamma1
        # self.Gamma2 = Gamma2

    def set_E_ph_symmetric(self, es, ts):
        assert len(es) == self.Nb, "es doesn't have size Nb"
        assert len(ts) == self.Nb, "ts doesn't have size Nb"

        t = np.array([*ts, *(ts[::-1])])
        E = np.array([*es, 0, *(es[::-1] * (-1))])
        offset = [-1, 0, 1]

        self.E = diags([t, E, t], offset)

    def set_E_full(self, es, ts):

        assert len(es) == 2 * self.Nb + 1
        assert len(ts) == 2 * self.Nb

        offset = [-1, 0, 1]

        self.E = diags([ts, es, ts], offset)

    def get_auxiliary_hybridization(self):
        pass
