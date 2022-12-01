import numpy as np

import pandas as pd
import torch
from numpy import NAN

a = np.array([[1, 2, 3], [NAN, 5, 6]])
rows, cols = np.where(np.isnan(a) == False)
b = a[rows, cols]
print(b)

