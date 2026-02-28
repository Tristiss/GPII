import numpy as np
import scipy.constants as const
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from numba import njit
from numba_progress import ProgressBar

@njit(nogil = True)
def simulation(numba_prog):
    n_h2o = const.c / 4 #1.33
    n_air = const.c / 1.00029

    s_min = 5
    s_max = 8
    t_min = 0
    t_max = 1e-07
    x_min = y_min = -10
    x_max = y_max = 10

    N_x = N_y = 500

    h = (x_max - x_min) / N_x
    dt = 1e-10

    x_index = np.arange(x_min, x_max, h)
    y_index = np.arange(y_min, y_max, h)

    u_db = []

    alpha = np.zeros((N_x, N_y), dtype = np.float64)

    for i in x_index:
        for j in y_index:
            if s_max >= j - np.sin(i) >= s_min:
                alpha[np.where(x_index == i)[0][0]][np.where(y_index == j)[0][0]] = n_h2o
            else:
                alpha[np.where(x_index == i)[0][0]][np.where(y_index == j)[0][0]] = n_air


    def wave_origin(t:np.float64):
        return np.sin(1e04 * t) * 200

    def update_meshgrid(x:int, y:int, u_1:np.ndarray, u_2:np.ndarray):
        a = alpha[x][y]
        return np.square(dt * a / h) * (u_1[x - 1][y] + u_1[x + 1][y] + u_1[x][y - 1] + u_1[x][y + 1] - 4 * u_1[x][y]) + 2 * u_1[x][y] - u_2[x][y]
        
    u_2 = np.zeros((N_x, N_y), dtype = np.float64)
    u_1 = np.zeros((N_x, N_y), dtype = np.float64)

    for i in x_index:
        for j in y_index[0:10]:
            u_2[np.where(x_index == i)[0][0]][np.where(y_index == j)[0][0]] = 200
            u_1[np.where(x_index == i)[0][0]][np.where(y_index == j)[0][0]] = 200

    u = np.zeros((N_x, N_y), dtype = np.float64)

    for t in np.arange(t_min, t_max, dt, dtype = np.float64):
        numba_prog.update(1)
        for i in x_index[1:-1]:
            for j in y_index[1:-1]:
                x = np.where(x_index == i)[0][0]
                y = np.where(x_index == j)[0][0]
                u[np.where(x_index == i)[0][0]][np.where(y_index == j)[0][0]] = update_meshgrid(x, y, u_1, u_2)
        
        for i in x_index:
            for j in y_index[0:10]:
               u[np.where(x_index == i)[0][0]][np.where(y_index == j)[0][0]] = wave_origin(t)

        u_db.append(u)

        u_2 = u_1
        u_1 = u
        u = np.zeros((N_x, N_y), dtype = np.float64)
    return u_db

t_min = 0
t_max = 1e-07
dt = 1e-10

with ProgressBar(total = len(np.arange(t_min, t_max, dt)), ncols = 80) as numba_prog_1:
    u_db = simulation(numba_prog_1)

fig, axs = plt.subplots()

im = plt.imshow(u_db[0], animated = True, interpolation = "none")

def update(frame):
    im.set_array(u_db[frame])
    return im,

anim = ani.FuncAnimation(fig, update, interval=50, blit=True, frames = len(u_db))
anim.save("Wave_sim_v4.gif", fps = 10, writer = "PillowWriter")
plt.show()