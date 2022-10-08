from typing import Union
import src.greens_function.frequency_greens_function as fg
import src.greens_function.time_greens_function as tg


class KeldyshIdentity:
    """Identity operator class for the Keldysh contour in the
    Keldysh rotated representation
    """

    def __init__(self) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """

    def __mul__(self, other: Union["fg.FrequencyGreen", "tg.TimeGreen", int,
                                   float, complex]
                ) -> Union["fg.FrequencyGreen", "tg.TimeGreen", complex]:
        """Defines multiplication of identity in Keldysh contour matrix with
        Green's function or a number.

        Returns
        -------
        out : FrequencyGreen, TimeGreen, int, float or complex
            Result of multiplication with identity
        """
        return other

    def __add__(self, other: Union["fg.FrequencyGreen", "tg.TimeGreen",
                                   "KeldyshIdentity"]
                ) -> Union["fg.FrequencyGreen", "tg.TimeGreen"]:
        """Returns result of addition of identity with Green's function.

        Returns
        -------
        FrequencyGreen
            Result of addition of identity with Green's function

        Raises
        ------
        ValueError
            If 'other' is not of type Identity, FrequencyGreen or TimeGreen.
        """
        if isinstance(other, fg.FrequencyGreen):
            return fg.FrequencyGreen(other.freq, other.retarded + 1,
                                     other.keldysh)
        elif isinstance(other, tg.TimeGreen):
            return tg.TimeGreen(other.time, other.retarded + 1,
                                other.keldysh)
        elif isinstance(other, KeldyshIdentity):
            return self
        else:
            raise ValueError('The object that is added has to be of type ' +
                             'Identity, FrequencyGreen or TimeGreen.')

    def __sub__(self, other: Union["fg.FrequencyGreen", "tg.TimeGreen"]
                ) -> Union["fg.FrequencyGreen", "tg.TimeGreen"]:
        """Returns result of subtracting a Green's function from the
        identity

        Returns
        -------
        FrequencyGreen
            Result of subtracting other from an identity matrix

        Raises
        ------
        ValueError
            If 'other' is not of type FrequencyGreen or TimeGreen.
        """
        if isinstance(other, fg.FrequencyGreen):
            return fg.FrequencyGreen(other.freq, 1 - 1 * other.retarded,
                                     -1 * other.keldysh)
        elif isinstance(other, tg.TimeGreen):
            return tg.TimeGreen(other.freq, 1 - 1 * other.retarded,
                                -1 * other.keldysh)
        else:
            raise ValueError('The object that is added has to be of type ' +
                             'FrequencyGreen or TimeGreen.')


Identity = KeldyshIdentity()
