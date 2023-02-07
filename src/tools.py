__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def simulate_trajectories(x_init, dynamical_system, dt=0.01, trajectory_length=4000):
    x_hist = [x_init]
    dx_hist =[x_init * 0]
    for i in range(trajectory_length):
        dx = dynamical_system.ds_stabilizer(x_hist[i])[0]
        x = x_hist[i] + dx * dt

        x_hist.append(x)
        dx_hist.append(dx)

    return x_hist, dx_hist


def load_saved_mat_file(data_name):
    """
        Loads a matlab file from a subdirectory.
    """

    dataset_path = 'data/lasa_handwriting_dataset'
    data = sio.loadmat(os.path.join(dataset_path, data_name + '.mat'))
    dataset = data['demos']
    num_demos = int(dataset.shape[1])
    demo_length = dataset[0][0]['pos'][0][0].shape[1]
    demoIdx = []
    demonstrations = np.empty([4, num_demos * demo_length])
    for i in range(num_demos):
        pos = dataset[0][i]['pos'][0][0]
        vel = dataset[0][i]['vel'][0][0]
        demonstrations[:2, i * demo_length:(i + 1) * demo_length] = pos
        demonstrations[2:, i * demo_length:(i + 1) * demo_length] = vel

        demoIdx.append(i * demo_length)

    return demonstrations, np.array(demoIdx), demo_length


def plot_results(data, data_name, demoIdx, demo_length, dynamical_system, plot_mode, extra=10, simulate_length=1500, n_points=100):
    plt.rcdefaults()
    plt.rcParams.update({"text.usetex": True, "font.family": "Times New Roman", "font.size": 26})
    plt.figure(figsize=(8, 8))

    plt.title('Lyapunov Learner (CLF-DM GMR)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

    x_lim = [[np.min(data[0, :]) - extra, np.max(data[0, :]) + extra],
             [np.min(data[1, :]) - extra, np.max(data[1, :]) + extra]]

    x0_all = data[:2, demoIdx]  # finding initial points of all demonstrations

    # Simulate trajectories
    x_sim, _ = simulate_trajectories(x0_all, dynamical_system, trajectory_length=simulate_length)

    # Plot simulated data
    x_sim = np.array(x_sim)
    plt.plot(x_sim[:, 0, :], x_sim[:, 1, :], color='red', linewidth=4, zorder=10)

    if plot_mode == 'velocities':
        # Plot demonstrations
        for i in range(int(data.shape[1] / demo_length)):
            plt.scatter(data[0, i * demo_length:(i + 1) * demo_length], data[1, i * demo_length:(i + 1) * demo_length],
                        color='white', zorder=5, alpha=0.5)

        # Plot goal
        plt.scatter(0, 0, linewidth=4, color='blue', zorder=13)  # goal is at zero in LASA dataset

        # Get velocities
        x1_coords, x2_coords = np.meshgrid(
            np.linspace(x_lim[0][0], x_lim[0][1], n_points),
            np.linspace(x_lim[1][0], x_lim[1][1], n_points))

        x_init = np.zeros([2, n_points ** 2])
        x_init[0, :] = x1_coords.reshape(-1)
        x_init[1, :] = x2_coords.reshape(-1)
        dx_hist = []
        for i in range(n_points ** 2):  # TODO: do this in one pass
            x, dx = simulate_trajectories(x_init[:, i].reshape(-1, 1), dynamical_system, trajectory_length=1)
            dx_hist.append(dx[1])

        dx_hist = np.array(dx_hist)[:, :, 0]

        vel = dx_hist.reshape(n_points, n_points, -1)
        norm_vel = np.clip(np.linalg.norm(vel, axis=2), a_min=0, a_max=50)

        # Plot vector field
        cmap = plt.cm.Greys
        color = 'black'
        plt.streamplot(
            x1_coords, x2_coords,
            vel[:, :, 0], vel[:, :, 1],
            color=color, cmap=cmap, linewidth=0.5,
            density=2, arrowstyle='fancy', arrowsize=1, zorder=2
        )

        # Plot speed
        CS = plt.contourf(x1_coords, x2_coords, norm_vel, cmap='viridis', levels=50, zorder=1)

        min_vel_ceil = np.ceil(np.min(norm_vel))
        max_vel_floor = np.ceil(np.max(norm_vel))
        delta_x = max_vel_floor / 10
        cbar = plt.colorbar(CS, location='bottom', ticks=np.arange(min_vel_ceil, max_vel_floor, delta_x))
        cbar.ax.set_xlabel('speed (mm/s)')

    elif plot_mode == 'energy_levels':
        # Plot demonstrations
        for i in range(int(data.shape[1] / demo_length)):
            plt.scatter(data[0, i * demo_length:(i + 1) * demo_length], data[1, i * demo_length:(i + 1) * demo_length],
                        color='blue', zorder=5, alpha=0.5)

        # Plot energy levels
        dynamical_system.lyapunov.energyContour(dynamical_system.Vxf, x_lim)
        plt.plot(0, 0, 'g*', markersize=15, linewidth=3, label='target', zorder=12)
    else:
        print('Selected print mode not valid!')
        exit()

    plt.xlim(x_lim[0])
    plt.ylim(x_lim[1])
    plt.tight_layout()
    plt.savefig('%s_vector_field_%s.pdf' % (data_name, plot_mode), dpi=300)
    plt.show()