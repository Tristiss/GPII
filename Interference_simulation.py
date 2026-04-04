import numba
import numpy as np
import pandas as pd
import scipy.constants as const
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from numba_progress import ProgressBar

n_h2o = np.float64(const.c / 1.33) # 1.33
n_air = np.float64(const.c / 1.00029)

surf_l = np.int32(0)
surf_r = np.int32(0)

s_min = np.int32(0)
s_max = np.int32(1)
x_min = y_min = np.int32(-10)
x_max = y_max = np.int32(10)

N_x = N_y = np.int32(750)
N_t = np.int32(14e03)

h = (x_max - x_min) / N_x

t_min = np.int32(0)
t_max = np.float64(7e-08) # 1.4e-07
dt = (t_max - t_min) / N_t
freq = np.int64(4e09) # 3e09

animation_res = 100

t_res = len(np.arange(t_min, t_max, dt))

x_index = np.arange(x_min, x_max, h)
y_index = np.arange(y_min, y_max, h)

x_y_res = len(x_index)

courant = n_h2o * dt / h
min_h_res = h / (n_h2o / (2 * freq))

print(f"Courant: {courant}, Frequency condition: {min_h_res}")
if courant >= 1 or min_h_res >= 1:
    raise ValueError("Simulation does not have the necessary resolution")

@numba.njit
def surface(i:np.int32, j:np.int32, surf:np.int32):
    match surf:
        case 0:
            return y_index[j] - np.sin(x_index[i]) # sine function
        case 1:
            return y_index[j] - np.cos(x_index[i]) # cosine function
        case 2:
            return y_index[j] - 0.05 * np.square(x_index[i]) # parabola (open to right)
        case 3:
            return y_index[j] + 0.05 * np.square(x_index[i]) # parabola (open to left)
        case 4:
            return np.sqrt(np.square(y_index[j]) - np.square(x_index[i])) # hyperbola
        case 5:
            return np.sqrt(np.square(y_index[j]) + np.square(x_index[i])) # circle
        case _:
            return 0
        
@numba.njit(fastmath = True, parallel = True)
def alpha_init():
# ----------------------- ALPHA INITIALISATION -----------------------
    alpha = np.zeros((N_x, N_y), dtype = np.float64)

    for i in numba.prange(x_y_res):
        for j in range(x_y_res):
            i = np.int32(i)
            j = np.int32(j)
            if surface(i, j, surf_l) >= s_min and s_max >= surface(i, j, surf_r):
                alpha[i, j] = np.square(dt * n_h2o / h)
            else:
                alpha[i, j] = np.square(dt * n_air / h)
    return alpha

alpha = alpha_init()

@numba.njit(fastmath = True, parallel = True, nogil = True)
def update_mesh(u, u_1, u_2):
    for i in numba.prange(x_y_res - 2):
        for j in range(x_y_res - 2):
            u[i + 1, j + 1] = alpha[i + 1, j + 1] * (u_1[i, j + 1] + u_1[i + 2, j + 1] + u_1[i + 1, j] + u_1[i + 1, j + 2] - 4 * u_1[i + 1, j + 1]) + 2 * u_1[i + 1, j + 1] - u_2[i + 1, j + 1]
    return u

@numba.njit(fastmath = True, parallel = False, nogil = True)
def simulation(numba_prog):
    # ----------------------- WAVE SIMULATION -----------------------
    u_2 = np.zeros((N_x, N_y), dtype = np.float64)
    u_1 = np.zeros((N_x, N_y), dtype = np.float64)
    tmp = np.zeros((N_x, N_y), dtype = np.float64)
    
    for i in range(x_y_res):
        for j in range(2):
            u_2[np.int32(i), np.int32(j)] = 200
            u_1[np.int32(i), np.int32(j)] = 200

    u = np.empty((N_x, N_y), dtype = np.float64)

    u_db = []

    for t, count in zip(np.arange(t_min, t_max, dt, dtype = np.float64), range(N_t)):
        numba_prog.update(1)

        u = update_mesh(u, u_1, u_2)

        for i in range(x_y_res):
            for j in range(2):
                u[i, j] = np.sin(freq * t) * 200

        if bool(count % animation_res) == False:
            u_db.append(np.copy(u))
        tmp = u_2
        u_2 = u_1
        u_1 = u
        u = tmp #np.empty((N_x, N_y), dtype = np.float64)
    return u_db

with ProgressBar(total = len(np.arange(t_min, t_max, dt)), ncols = 80) as numba_prog_1:
    u_db = simulation(numba_prog_1)

#df = pd.DataFrame(u_db[-40])

#df.to_csv("Sim_data_sinsin.csv", sep = ";")

fig_x, axs_x = plt.subplots()

axs_x.imshow(u_db[-40])

fig, axs = plt.subplots()

im = plt.imshow(u_db[0], animated = True, interpolation = "none")

def update(frame):
    im.set_array(u_db[frame])
    return im,

anim = ani.FuncAnimation(fig, update, interval = 50, blit = True, frames = len(u_db))
#anim.save("Wave_sim_v7.gif", fps = 10, writer = "Pillow")
plt.show()