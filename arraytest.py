import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import read
from scipy.signal import find_peaks
from scipy.signal import medfilt
import scipy.stats.stats as st

x = [None,1,2,3,None,0]



for i in range(len(x)):
    if(x[i] != None and x[i] != 0):
        print(x[i])



print(x)