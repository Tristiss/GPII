import numba
import numpy as np
import scipy.constants as const
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from numba_progress import ProgressBar

n_h2o = np.float64(const.c / 1.33) # 1.33
n_air = np.float64(const.c / 1.00029)

surf_l = np.int64(4)
surf_r = np.int64(5)

s_min = np.float64(4)
s_max = np.float64(20)
x_min = y_min = np.float64(-20)
x_max = y_max = np.float64(20)

N_x = N_y = np.int64(750)

h = (x_max - x_min) / N_x

t_min = np.float64(0)
t_max = np.float64(1.4e-07)
dt = np.float64(1e-11)
freq = np.float64(3e9)

t_res = len(np.arange(t_min, t_max, dt))

x_index = np.arange(x_min, x_max, h)
y_index = np.arange(y_min, y_max, h)

x_y_res = len(x_index)

@numba.njit(fastmath = True)
def surface(i:np.int64, j:np.int64, surf:np.int64):
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

@numba.njit(fastmath = True)
def alpha_init():
    alpha = np.zeros((N_x, N_y), dtype = np.float64)

    for i in numba.prange(x_y_res):
        for j in range(x_y_res):
            i = np.int64(i)
            j = np.int64(j)
            if surface(i, j, surf_l) >= s_min and s_max >= surface(i, j, surf_r):
                alpha[i][j] = n_h2o
            else:
                alpha[i][j] = n_air
    return alpha
"""
@numba.njit(fastmath = True)
def beta_init():
    beta = np.zeros((N_x, N_y), dtype = np.float64)

    for i in numba.prange(x_y_res):
        for j in range(x_y_res):
            beta[i][j] = ((1 / (x_index[i] + 11)) - (1 / (x_index[i] - 11)) - 1.4) * (- (1 / (y_index[j] - 11)) - 1.4)
    return beta
"""
alpha = alpha_init()
#beta = beta_init()

@numba.njit(fastmath = True)
def wave_origin(t:np.float64):
        return np.sin(freq * t) * 200

@numba.njit(fastmath = True)
def update_meshgrid(x:np.int64, y:np.int64, u_1:np.ndarray, u_2:np.ndarray):
    a = alpha[x][y]
    return np.square(dt * a / h) * (u_1[x - 1][y] + u_1[x + 1][y] + u_1[x][y - 1] + u_1[x][y + 1] - 4 * u_1[x][y]) + 2 * u_1[x][y] - u_2[x][y]

@numba.njit(nogil = True, fastmath = True)
def simulation(numba_prog):
    u_db = []
        
    u_2 = np.zeros((N_x, N_y), dtype = np.float64)
    u_1 = np.zeros((N_x, N_y), dtype = np.float64)

    for i in numba.prange(x_y_res - 10):
        for j in range(10):
            u_2[np.int64(i + 5)][np.int64(j)] = 200
            u_1[np.int64(i + 5)][np.int64(j)] = 200

    u = np.zeros((N_x, N_y), dtype = np.float64)

    counter = 0

    for t in np.arange(t_min, t_max, dt, dtype = np.float64):
        numba_prog.update(1)
        for i in numba.prange(x_y_res - 2):
            for j in range(x_y_res - 2):
                x = np.int64(i + 1)
                y = np.int64(j + 1)
                u[i + 1][j + 1] = update_meshgrid(x, y, u_1, u_2)
        
        for i in numba.prange(x_y_res - 10):
            for j in range(10):
                u[np.int64(i + 5)][np.int64(j)] = wave_origin(t)

        if counter > 20:
           u_db.append(u)
           counter = 0
        counter += 1
        u_2 = u_1
        u_1 = u
        u = np.zeros((N_x, N_y), dtype = np.float64)
    return u_db

with ProgressBar(total = len(np.arange(t_min, t_max, dt)), ncols = 80) as numba_prog_1:
    u_db = simulation(numba_prog_1)

fig, axs = plt.subplots()

im = plt.imshow(u_db[0], animated = True, interpolation = "none")

def update(frame):
    im.set_array(u_db[frame])
    return im,

anim = ani.FuncAnimation(fig, update, interval = 50, blit = True, frames = len(u_db))
#anim.save("Wave_sim_v7.gif", fps = 10, writer = "Pillow")
plt.show()