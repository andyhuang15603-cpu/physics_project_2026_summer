import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

Omega = 2 * np.pi * 0.5
Theta = np.deg2rad(30)
t_max = 12.0
frames = int(25 * t_max)

dt = 1 / 25
t = np.linspace(0, t_max, frames)

sx_all = np.sin(Theta) * np.cos(Omega * t)
sy_all = np.sin(Theta) * np.sin(Omega * t)
sz_all = np.cos(Theta) * np.ones_like(t)

fig = plt.figure(figsize = (12, 10))
gs = fig.add_gridspec(2, 2, wspace = 0.3, hspace = 0.3)

ax = fig.add_subplot(gs[0, 0], projection='3d') 
ax_sx = fig.add_subplot(gs[0, 1])
ax_sy = fig.add_subplot(gs[1, 0])
ax_sz = fig.add_subplot(gs[1, 1]) 

ax.view_init(elev = 25, azim = 45)
ax.set_xlabel('$S_x$', fontsize = 12)
ax.set_ylabel('$S_y$', fontsize = 12)
ax.set_zlabel('$S_z$', fontsize = 12)
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_zlim([-1.1, 1.1])
ax.set_title("Larmor Precession on Bloch Sphere")

u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color = 'gray', alpha = 0.1, lw = 0, antialiased = True)

ax.plot([-1.1, 1.1], [0, 0], [0, 0], color = 'black', alpha = 0.4, ls = '--')
ax.plot([0, 0], [-1.1, 1.1], [0, 0], color = 'black', alpha = 0.4, ls = '--')
ax.plot([0, 0], [0, 0], [-1.1, 1.1], color = 'black', alpha = 0.4, ls = '--')

trail_line, = ax.plot([], [], [], color = 'purple', alpha = 0.7, lw = 2)

axes_2d = [ax_sx, ax_sy, ax_sz]
y_datasets = [sx_all, sy_all, sz_all]
labels = ['$S_x(t)$', '$S_y(t)$', '$S_z(t)$']
colors = ['green', 'blue', 'orange']

for a_2d, data, label, color in zip(axes_2d, y_datasets, labels, colors):
    a_2d.set_xlim([0, t_max])
    a_2d.set_ylim([-1.2, 1.2])
    a_2d.set_xlabel('Time ($t$)', fontsize = 10)
    a_2d.set_ylabel(label, fontweight = 'bold', color = color)
    a_2d.grid(True, alpha = 0.3)
    # Optional: Plots a faint baseline guide so the user sees the final curve path
    a_2d.plot(t, data, color = color, alpha = 0.15, ls = '--')

line_sx, = ax_sx.plot([], [], color = 'green', lw = 2)
dot_sx,  = ax_sx.plot([], [], color = 'green', marker = 'o', ms = 6)

line_sy, = ax_sy.plot([], [], color = 'blue', lw = 2)
dot_sy,  = ax_sy.plot([], [], color = 'blue', marker = 'o', ms = 6)

line_sz, = ax_sz.plot([], [], color = 'orange', lw = 2)
dot_sz,  = ax_sz.plot([], [], color = 'orange', marker = 'o', ms = 6)

current_artists = []

def update(frame):
    global current_artists
    for artist in current_artists:
        artist.remove()
    current_artists.clear()

    current_t = t[frame]
    x, y, z = sx_all[frame], sy_all[frame], sz_all[frame]

    trail_line.set_data(sx_all[: frame + 1], sy_all[: frame + 1])
    trail_line.set_3d_properties(sz_all[: frame + 1])

    spin_arrow = ax.quiver(0, 0, 0, x, y, z, color = 'red', lw = 3, arrow_length_ratio = 0.1)
    current_artists.append(spin_arrow)

    proj_x, = ax.plot([x, x], [0, y], [0, z], color = 'green', ls = ':', alpha = 0.6)
    proj_y, = ax.plot([0, x], [y, y], [0, z], color = 'blue', ls = ':', alpha = 0.6)
    proj_z, = ax.plot([0, x], [0, y], [z, z], color = 'orange', ls = ':', alpha = 0.6)
    
    pt_x = ax.scatter([x], [0], [0], color = 'green', s = 20)
    pt_y = ax.scatter([0], [y], [0], color = 'blue', s = 20)
    pt_z = ax.scatter([0], [0], [z], color = 'orange', s = 20)

    current_artists.extend([proj_x, proj_y, proj_z, pt_x, pt_y, pt_z])

    line_sx.set_data(t[: frame + 1], sx_all[: frame + 1])
    dot_sx.set_data([current_t], [x])

    line_sy.set_data(t[: frame + 1], sy_all[: frame + 1])
    dot_sy.set_data([current_t], [y])

    line_sz.set_data(t[: frame + 1], sz_all[: frame + 1])
    dot_sz.set_data([current_t], [z])

    return [trail_line, line_sx, dot_sx, line_sy, dot_sy, line_sz, dot_sz] + current_artists

ani = FuncAnimation(fig, update, frames = frames, interval = 30, blit = False, repeat = True)
plt.show()
