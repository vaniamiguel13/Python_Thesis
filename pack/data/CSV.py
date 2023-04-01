import pandas as pd
import numpy as np
from IPython.core.display_functions import display


def read_csv(path):
    data = pd.read_csv(path, sep=';', header=0)
    return data
