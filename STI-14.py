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

alpha_k = [0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173]

beta_k = [0.085, 0.078, 0.065, 0.011, 0.047, 0.095]

# ---------- System Variables ----------

path = r"C:\Programmieren\Praktikum\GPII\Data"

config = {
    "duration": 10, # duration of the genererated signal in s
    "dead_time": 1,
    "srate": 9600, # sample rate in Hz
    "cal_amp": 1,
    "cal_freq": 300.0,
    "n_samples": len(k_vals) * len(mod_vals), # add additional samples if more than STI-14 is performed
}

config["time"] = np.arange(0, config["duration"], 1 / config["srate"])


# ---------- Signal Generation ----------

def signal_generation(config:dict):
    """
    start of signal is silence, calibration, silence
    """
    @numba.njit(fastmath = True, parallel = True)
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

        for i in numba.prange(len(frex)):
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

    silence = np.array([0 for i in range(config["dead_time"] * config["srate"])])
    calibration = np.array([config["cal_amp"] * np.sin(2 * np.pi * config["cal_freq"] * t/config["srate"]) for t in range(int(config["srate"]//config["cal_freq"]))])

    full_signal = silence.copy()
    full_signal = np.concatenate((full_signal, calibration), axis = 0)
    full_signal = np.concatenate((full_signal, silence), axis = 0)

    for k in tqdm(k_vals):
        for m in tqdm(mod_vals):
            noise = pink_noise_v2(k_vals[k]["f_c"], config["time"])
            sign = am_modulation(noise, m, config["time"])
            sign = G_k(sign, k)
            full_signal = np.concatenate((full_signal, sign), axis = 0)
            full_signal = np.concatenate((full_signal, silence), axis = 0)
    return full_signal

# ---------- Measurement Pipeline ----------

def measurement(sign:np.ndarray): # needs to be implemented
    return [sign, sign] # first ref then mes signal

# ---------- Intermediary preparation of the measurement data ----------

def signal_slicing(sign:np.ndarray, config:dict):
    peaks, props = find_peaks(sign)

    silence = config["dead_time"] * config["srate"]
    sample_size = config["duration"] * config["srate"]
    peak_width = int(3/4 * config["srate"]/config["cal_freq"] + 23)

    sign = sign[peaks[0]:]

    sliced_signals = []
    for i in range(config["n_samples"]):
        low_index = silence + peak_width + silence * i + sample_size * i
        high_index = silence + peak_width + silence * i + sample_size * (i+1)
        sliced_signals.append(sign[low_index : high_index]) 

    arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)
    counter = 0
    for k in range(len(k_vals)):
        for f_m in range(len(mod_vals)):
            arr[k, f_m] = sliced_signals[counter]
            counter += 1
    return arr

# ---------- STI Computation ----------

def sti_comp(signs, config:dict):
    def envelope_detection(sign:np.ndarray, srate:int):
        arr = np.empty(shape = (len(k_vals), len(mod_vals)), dtype=np.ndarray)
        for i_k in range(len(k_vals)):
            for j_f_m in range(len(mod_vals)):
                sign[i_k, j_f_m] *= sign[i_k, j_f_m]
                sos = signal.butter(20, 100, 'low', fs = srate, output = "sos")
                y = signal.sosfilt(sos, sign[i_k, j_f_m])
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
    
    def limit_mod_ratio(m):
        return min([m, 1])

    def snr_comp(m):
        if m == 1:
            return 15
        snr = 10 * np.log10(m / (1 - m))
        match snr:
            case tmp if snr < -15:
                return -15
            case tmp if snr > 15:
                return 15
            case _:
                return snr
    
    def transmission_index(snr):
        return (snr + 15) / 30
    
    def modulation_transfer_index(ti):
        mti = np.empty(len(k_vals))
        for k in range(len(k_vals)):
            mti[k] = np.mean(ti[k])
        return mti

    def sti_last_step(mti):
        first_term = np.sum([alpha_k[k] * mti[k] for k in range(len(k_vals))])
        second_term = np.sum([beta_k[k] * np.sqrt(mti[k] * mti[k+1]) for k in range(len(k_vals)-1)])
        return first_term - second_term

    params = {
        "sign": [],
        "I_k_m": [],
        "mod_dep": []
    }

    for sign, i in zip(signs, range(len(signs))):
        params["sign"].append(signal_slicing(sign, config))
        params["I_k_m"].append(envelope_detection(params["sign"][i], config["srate"]))
        params["mod_dep"].append(modulation_depths(params["I_k_m"][i], config["time"]))

    m_k_fm = params["mod_dep"][1] / params["mod_dep"][0]

    limit_mod_ratio_vec = np.vectorize(limit_mod_ratio)
    m_k_fm = limit_mod_ratio_vec(m_k_fm)

    # steps 5 and 6 are still missing because we first need to understand what value to use for I_k

    snr_comp_vec = np.vectorize(snr_comp)
    snr_k_fm = snr_comp_vec(m_k_fm)

    transmission_index_vec = np.vectorize(transmission_index)
    ti = transmission_index_vec(snr_k_fm)

    mti = modulation_transfer_index(ti)

    sti = sti_last_step(mti)
    print(sti)

    return sti, ti

# ---------- Main ----------

def main():
    sign = signal_generation(config)

    signs = measurement(sign)

    # ---------- Saving the Data in a csv file ----------

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
            pd.DataFrame({"input": sign, "ref": signs[0], "mes": signs[1]}).to_csv(newpath + r"\Messdaten.csv", sep = ";")
            break

    sti, ti = sti_comp(signs, config)
    plt.imshow(ti)
    plt.show()

if __name__ == "__main__":
    main()
