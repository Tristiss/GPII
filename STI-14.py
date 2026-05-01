import numba
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from os import makedirs
from scipy import signal
from numpy import random
from scipy.signal import find_peaks

#from Praktikumsmodul import *

# ---------- Variables prescribed by the standard ----------

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

# ---------- Signal Generation ----------

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

def G_k(sign:np.ndarray, k:int):
    return sign * 10**(k_vals[k]["L_k"] / 20)

def signal_generation(time:np.ndarray, dead_time, srate:int, cal_amp:float, cal_freq):
    """
    start of signal is silence, calibration, silence
    """
    silence = np.array([0 for i in range(dead_time * srate)])
    calibration = np.array([cal_amp * np.sin(2 * np.pi * cal_freq * t) for t in range(cal_freq/4 * srate)])
    
    full_signal = silence.copy()
    full_signal = calibration.copy()
    full_signal = silence.copy()

    for k in tqdm(k_vals):
        for m in mod_vals[0:1]:
            noise = pink_noise_v2(k_vals[k]["f_c"], time)
            sign = am_modulation(noise, m, time)
            sign = G_k(sign, k)
            full_signal = np.concatenate((full_signal, sign), axis = 0)
            full_signal = np.concatenate((full_signal, silence), axis = 0)
    return full_signal

# ---------- Measurement Pipeline ----------

def measurement(sign:np.ndarray): # needs to be implemented
    return [sign, sign] # first ref then mes signal

# ---------- Intermediary preparation of the measurement data ----------

def signal_slicing(sign:np.ndarray, srate:int, sample_time:float, dead_time:float, n_samples:int, cal_freq:float):
    peaks, props = find_peaks(sign, rel_height = 1)

    sign = sign[peaks[0] + srate * cal_freq / 4:]

    sliced_signals = [sign[sample_time * srate * (i-1) + dead_time * srate:sample_time * srate * i + dead_time * srate] for i in range(n_samples)]

    arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)

    counter = 0

    for k in range(len(k_vals)):
        for f_m in range(len(mod_vals)):
            arr[k, f_m] = sliced_signals[counter]
            counter += 1
    return arr

# ---------- STI Computation ----------

def envelope_detection(sign:np.ndarray, srate:int):
    arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)
    for i_k in range(len(k_vals)):
        for j_f_m in range(len(mod_vals)):
            sign[i_k, j_f_m] *= sign[i_k, j_f_m]
            sos = signal.butter(20, 100, 'low', fs = srate, output = "sos")
            y, zf = signal.sosfilt(sos, sign[i_k, j_f_m])
            arr[i_k, j_f_m] = y
    return arr

def modulation_depths(I:np.ndarray, time:np.ndarray):
    arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)
    for i_k in range(len(k_vals)):
        for j_f_m in range(len(mod_vals)):
            sin_sum = (np.sum(I[i_k, j_f_m] * np.sin(2 * np.pi * mod_vals[j_f_m] * time)))**2
            cos_sum = (np.sum(I[i_k, j_f_m] * np.cos(2 * np.pi * mod_vals[j_f_m] * time)))**2
            denom_sum = np.sum(I[i_k, j_f_m])
            arr[i_k, j_f_m] = 2 * np.sqrt(sin_sum + cos_sum) / denom_sum
    return arr

def main():
    path = r"C:\Programmieren\Praktikum\GPII\Data"

    duration:float = 10 # duration of the genererated signal in s
    dead_time:float = 1
    srate:int = 44100 # sample rate in Hz
    cal_amp:float = 1
    cal_freq:float = 8e03

    n_samples:int = len(k_vals) + len(mod_vals) # add additional samples if more than STI-14 is performed

    time = np.arange(0, duration, 1 / srate)

    sign = signal_generation(time, dead_time, srate, cal_amp, cal_freq)

    signs = measurement(sign)

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
            pd.DataFrame({"time": time, "input": sign, "ref": signs[0], "mes": signs[1]}).to_csv(newpath + r"\Messdaten.csv", sep = ";")
            break

    for sign in signs:
        sign = signal_slicing(sign, srate, duration, dead_time, n_samples, cal_freq)
        I_k_m = envelope_detection(sign, srate)
        mod_dep = modulation_depths(I_k_m, time)


if __name__ == "__main__":
    main()
