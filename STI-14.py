import numba
import os.path
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
import pandas as pd
from tqdm import tqdm
from os import makedirs

from Praktikumsmodul import *

k_vals = {
    0 : {"f_c":125, "L_k": -2.5},
    1 : {"f_c":250, "L_k": 0.5},
    2 : {"f_c":500, "L_k": 0},
    3 : {"f_c":1000, "L_k": -6},
    4 : {"f_c":2000, "L_k": -12},
    5 : {"f_c":4000, "L_k": -18},
    6 : {"f_c":8000, "L_k": -24}
}

mod_vals = [0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5]

@numba.njit(fastmath = True, parallel = False)
def pink_noise_v2(f_c:float, time:np.ndarray):
    """
    f_c:float -> center frequency of the octave band
    time:np.ndarray -> 1d time array
    
    credit: Mike X Cohen, implemented from: https://www.youtube.com/watch?v=oKFSvwAbDhU&t=5s
    """

    low_f:float = f_c / np.sqrt(2)
    high_f:float = np.sqrt(2) * f_c 

    pnts = len(time)

    frex = np.linspace(low_f, high_f, 500)
    noise = np.zeros(pnts, dtype = np.float64)

    for i in range(len(frex)):
        amp = 1/(frex[i]**1)
        phase = random.random() * 2 * np.pi
        noise += amp * np.sin(2 * np.pi * frex[i] * time + phase)
    noise = noise - np.mean(noise)
    noise = noise / np.max(np.abs(noise))
    return noise

def am_modulation(sign:np.ndarray, f_m:float, time:np.ndarray):
    """
    sign:np.ndarray -> 1d signal array
    f_m:float -> amplitude modulation frequency
    time:np.ndarray -> 1d time array
    """
    amod = np.sqrt(0.5 * (1 + np.cos(2 * np.pi * f_m * time)))

    return sign * amod

def G_k(sign:np.ndarray, k):
    return sign * 10**(k_vals[k]["L_k"] / 20)

def signal_generation():
    duration:float = 10 # duration of the genererated signal in s
    dead_time:float = 1
    srate:int = 44100 # sample rate in Hz

    time = np.arange(0, duration, 1 / srate)

    silence = np.array([0 for i in range(dead_time * srate)])
    full_signal = silence.copy()

    for k in tqdm(k_vals):
        for m in mod_vals[0:1]:
            noise = pink_noise_v2(k_vals[k]["f_c"], time)
            sign = am_modulation(noise, m, time)
            sign = G_k(sign, k)
            full_signal = np.concatenate((full_signal, sign), axis = 0)
            full_signal = np.concatenate((full_signal, silence), axis = 0)
    return full_signal

def measurement(sign:np.ndarray): # needs to be implemented
    return sign, sign

def envelope_detection(sign):
    sign *= sign
    

def main():
    path = r"C:\Programmieren\Praktikum\GPII\Data"

    sign = signal_generation()
    mes_sign, ref_sign = measurement(sign)

    # Source - https://stackoverflow.com/questions/1274405/how-to-create-new-folder
    # Posted by mcandre, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-12-15, License - CC BY-SA 3.0
    #newpath = r'C:\Program Files\arbitrary' 
    #if not os.path.exists(newpath):
    #    os.makedirs(newpath)

    # check wether or not the path to the raw measurement data exists
    for i in range(100):
        newpath = path + rf"\Messung_{i}" 
        if not os.path.exists(newpath):
            makedirs(newpath)
            pd.DataFrame({"time": time, "input": sign, "ref": ref_sign, "mes": mes_sign}).to_csv(newpath + r"\Messdaten.csv", sep = ";")



if __name__ == "__main__":
    main()
