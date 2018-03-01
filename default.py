import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import mne

# import scipy.io

from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
import permute
from permute.core import two_sample

import random
# from random import sample
import math
import collections
from collections import defaultdict
from collections import OrderedDict
from collections import Counter
from collections import namedtuple
import itertools
import re
import time
import sys
import os
import warnings

import ipdb


ids = pd.IndexSlice
td = pd.Timedelta

warnings.filterwarnings("ignore")

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

tableau10 = [ (31, 119, 180),(255, 127, 14), (44, 160, 44),(214, 39, 40),
              (148, 103, 189), (140, 86, 75),
             (127, 127, 127), (23, 190, 207), (188, 189, 34), (227, 119, 194)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.) 
    
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau10)):  
    r, g, b = tableau10[i]  
    tableau10[i] = (r / 255., g / 255., b / 255.) 