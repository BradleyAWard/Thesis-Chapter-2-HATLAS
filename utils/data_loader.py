# --------------------------------------------------
# Imports
# --------------------------------------------------

import os
from astropy.table import Table
from pandas import read_csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# --------------------------------------------------
# Data loader
# --------------------------------------------------

def full_loader(file_name):
    """
    Loads a database

    :param file_name: Name of the data file located in 'data'
    :return: data
    """
    file_path = ROOT + '/data/' + file_name
    data = read_csv(file_path)

    return data

# --------------------------------------------------
# Data saver
# --------------------------------------------------

def save_file(data, file_name, R):
    """
    Saves a database

    :param file_name: Name of the data file located in 'data'
    :param R: The search radius for distinguishing file names
    :return: data
    """

    t_data = Table.from_pandas(data)
    file_path = ROOT + '/data/' + file_name
    t_data.write(file_path + '_' + '{}'.format(R), overwrite=True, format='csv')