# --------------------------------------------------
# Imports
# --------------------------------------------------

import numpy as np
import math as m
from tqdm import tqdm

# --------------------------------------------------
#  Counting Blanks (Fleuren et al., 2012)
# --------------------------------------------------

def blanks(data, r_limit, groupid: str, distance: str):
    """
    Blanks method (Fleuren et al.,) for a given r

    :param data: Input data file
    :param r_limit: The maximum radius to search for blank sources
    :param groupid: String for the counterpart group's ID
    :param distance: String for the separation column [arcsec]
    :return: blank_total
    """

    smallest_r = data.groupby([groupid])[distance]
    data = data.assign(min_r=smallest_r.transform(min))

    blank0 = len([r for idx, r in zip(data[groupid], data[distance]) if (m.isnan(idx)) & (r == 0)])
    blank1 = len([r for idx, r in zip(data[groupid], data[distance]) if (m.isnan(idx)) & (r != 0) & (r > r_limit)])
    blankmulti = len(
        np.unique([idx for idx, r in zip(data[groupid], data['min_r']) if (m.isnan(idx) == False) & (r > r_limit)]))
    blank_total = blank0 + blank1 + blankmulti

    return blank_total

# --------------------------------------------------

def blanks_gals(data, r_limit, groupid: str, distance: str, flag_SG: str):
    """
    Blanks method (Fleuren et al.,) for a given r, applied to galaxies

    :param data: Input data file
    :param r_limit: The maximum radius to search for blank sources
    :param groupid: String for the counterpart group's ID
    :param distance: String for the separation column [arcsec]
    :param flag_SG: String for the star/galaxy classification column
    :return: blank_total
    """

    galaxies = data[data[flag_SG] == 1]
    smallest_r = galaxies.groupby([groupid])[distance]
    data = data.assign(min_r_gal=smallest_r.transform(min))

    blank0 = len([r for idx, r in zip(data[groupid], data[distance]) if (m.isnan(idx)) & (r == 0)])

    blank1 = len([r for idx, r, flag in zip(data[groupid], data[distance], data[flag_SG]) if ((m.isnan(idx)) & (r != 0) & (flag == 1) & (r > r_limit)) | ((m.isnan(idx)) & (r != 0) & (flag == 0))])
    blankmulti = len(
        np.unique([idx for idx, r in zip(data[groupid], data['min_r_gal']) if (m.isnan(idx) == False) & (r > r_limit)]))
    blank_total = blank0 + blank1 + blankmulti

    return blank_total

# --------------------------------------------------

def B(r, sigma, Q0):
    """
    Functional form of blanks as a function of r

    :param r: Radial separation [arcsec]
    :param sigma: Positional offset error [arcsec]
    :param Q0: Fraction of sources beyond the limiting magnitude of the crossmatched survey
    :return: B function
    """

    f = 1 - np.exp(-(r**2)/(2*(sigma**2)))
    return 1 - Q0*f

# --------------------------------------------------
#  Positional Uncertainty
# --------------------------------------------------

def K(data, fwhm, sigma, f250: str, e250: str):
    """
    Determines the set of k-constants from a set of sources

    :param data: Input data file
    :param fwhm: Full Width at Half Maximum of detections
    :param sigma: Positional offset error [arcsec]
    :param f250: String for the 250-micron flux column [Jy]
    :param e250: String for the 250-micron flux error column [Jy]
    :return: k_values
    """

    k_values = []
    for obj in tqdm(range(len(data)), desc='Calculating K-constant values'):
        snr = data[f250][obj]/data[e250][obj]
        k_values.append(sigma*snr/fwhm)

    return k_values
