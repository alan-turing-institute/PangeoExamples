"""
These are different functions describing how labour is affected by increased WBGT.
I recommend reading doi.org/10.2760/07911
"""
import numpy as np

SQRT2 = np.sqrt(2)


def labour_sahu(WBGT):
    """Labour loss according to Sahu et al doi.org/10.2486/indhealth.2013-0006
    This is based on observations of rice harvesting.
    Is quite optimistic, 100% loss occurs at a very high WBGT.
    """
    return np.clip(100 - ((-5.14 * WBGT) + 218), 0, 100)


def labour_li(WBGT):
    """
    Labour loss function from Li et al doi.org/10.1016/j.buildenv.2015.09.005
    It quite optimistic, 100% loss occurs at a very high WBGT.
    """
    return np.clip(100 - ((-0.57 * WBGT) + 106.16), 0, 100)


def labour_dunne(WBGT):
    """
    Labour loss function from Dunne et al doi.org/10.1038/nclimate1827
    Based on safe working standards, so somewhat pessimistic.
    """
    return np.clip(100 - (100 - (25 * (np.maximum(0, WBGT - 25)) ** (2 / 3))), 0, 100)


def labour_hothaps_high(WBGT):
    """
    From https://www.sciencedirect.com/science/article/pii/S0959378019311306?via%3Dihub#sec0025
    """
    return 100 - 100 * (0.1 + (0.9 / (1 + (WBGT / 30.94) ** 16.64)))


def labour_hothaps_med(WBGT):
    """
    From https://www.sciencedirect.com/science/article/pii/S0959378019311306?via%3Dihub#sec0025
    """
    return 100 - 100 * (0.1 + (0.9 / (1 + (WBGT / 32.93) ** 17.81)))


def labour_hothaps_low(WBGT):
    """
    From https://www.sciencedirect.com/science/article/pii/S0959378019311306?via%3Dihub#sec0025
    """
    return 100 - 100 * (0.1 + (0.9 / (1 + (WBGT / 34.64) ** 22.72)))
