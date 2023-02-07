__credits__  	= 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ 	= 'MIT'

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


from cost.cost import Cost
from config import Vxf0, options
from utils.utils import guess_init_lyap
from stabilizer.ds_stab import dsStabilizer
from gmm.gmm import GMM


def load_gmm_parameters(x):
    matFile = sio.loadmat(x)

    Priors_EM, Mu_EM, Sigma_EM = matFile['Priors_EM'], matFile['Mu_EM'], matFile['Sigma_EM']
    return Priors_EM, Mu_EM, Sigma_EM


def simulate_trajectories(x_init, Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost, dt=0.01, trajectory_length=4000):
    x_hist = [x_init]
    dx_hist =[x_init * 0]
    for i in range(trajectory_length):
        dx = dsStabilizer(x_hist[i], Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost)[0]
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
    demo_length = 1000
    demoIdx = []
    demonstrations = np.empty([4, num_demos * demo_length])
    for i in range(num_demos):
        pos = dataset[0][i]['pos'][0][0]
        vel = dataset[0][i]['vel'][0][0]
        demonstrations[:2, i * demo_length:(i + 1) * demo_length] = pos
        demonstrations[2:, i * demo_length:(i + 1) * demo_length] = vel

        demoIdx.append(i * demo_length)

    return demonstrations, np.array(demoIdx)


def main(Vxf0, options):
    data_name = 'PShape'
    data, demoIdx = load_saved_mat_file(data_name)

    Vxf0['d'] = int(data.shape[0]/2)
    Vxf0.update(Vxf0)

    Vxf0 = guess_init_lyap(data, Vxf0, options['int_lyap_random'])
    cost = Cost()

    while cost.success:
        print('Optimizing the lyapunov function')
        Vxf, J = cost.learnEnergy(Vxf0, data, options)
        if cost.success:
            print('optimization succeeded without violating constraints')
            break

    extra = 10

    x_lim = [[np.min(data[0, :]) - extra, np.max(data[0, :]) + extra],
                   [np.min(data[1, :]) - extra, np.max(data[1, :]) + extra]]

    # x_lim = [[0, 0], [0, 0]]
    # x_lim[0][0] = -15.1  # S: -15.1; P:-32.22; DB:-41.03
    # x_lim[0][1] = 51.4  # S: 51.4; P: 28.51; DB: 24.37
    # x_lim[1][0] = -7.73  # S: -7.73; P: -28.16; DB: -22.71
    # x_lim[1][1] = 56  # S: 56; P: 37.43; DB: 9.45

    # h3 = cost.energyContour(Vxf, axes_limits, np.array(()), np.array(()), np.array(()), False)
    # h2 = plt.plot(0, 0, 'g*', markersize=15, linewidth=3, label='target')
    plt.rcdefaults()
    plt.rcParams.update({"text.usetex": True, "font.family": "Times New Roman", "font.size": 26})
    plt.figure(figsize=(8, 8))

    plt.title('Lyapunov Learning (CLF-DM GMR) ')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

    # Run DS
    opt_sim = dict()
    opt_sim['dt'] = 0.01
    opt_sim['i_max'] = 4000
    opt_sim['tol'] = 1

    d = data.shape[0]/2  # dimension of data
    x0_all = data[:int(d), demoIdx]  # finding initial points of all demonstrations

    # get gmm params
    gmm = GMM(num_clusters=options['num_clusters'])
    gmm.update(data.T, K=options['num_clusters'], max_iterations=100)
    mu, sigma, priors = gmm.mu, gmm.sigma, gmm.logmass
    mu = mu.T
    sigma = sigma.T
    priors = priors.T


    # rho0 and kappa0 impose minimum acceptable rate of decrease in the energy
    # function during the motion. Refer to page 8 of the paper for more information
    rho0 = 1
    kappa0 = 0.1

    inp = list(range(Vxf['d']))
    output = np.arange(Vxf['d'], 2 * Vxf['d'])

    xd, _ = dsStabilizer(x0_all, Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost)

    # Simulate trajectories
    x_sim, _ = simulate_trajectories(x0_all, Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost, trajectory_length=1500)

    # Plot simulated data
    x_sim = np.array(x_sim)
    plt.plot(x_sim[:, 0, :], x_sim[:, 1, :], color='red', linewidth=4, zorder=10)


    # Plot demonstrations
    demo_length = 1000
    for i in range(int(data.shape[1] / demo_length)):
        plt.scatter(data[0, i * demo_length:(i + 1) * demo_length], data[1, i * demo_length:(i + 1) * demo_length],
                 color='white', zorder=5, alpha=0.5)

    # Get velocities
    n_points = 100

    x1_coords, x2_coords = np.meshgrid(
        np.linspace(x_lim[0][0], x_lim[0][1], n_points),
        np.linspace(x_lim[1][0], x_lim[1][1], n_points))

    x_init = np.zeros([2, n_points ** 2])
    x_init[0, :] = x1_coords.reshape(-1)
    x_init[1, :] = x2_coords.reshape(-1)
    dx_hist = []
    for i in range(n_points ** 2):
        print(i)
        x, dx = simulate_trajectories(x_init[:, i].reshape(-1, 1), Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost, trajectory_length=1)
        dx_hist.append(dx[1])

    dx_hist = np.array(dx_hist)[:, :, 0]

    vel = dx_hist.reshape(n_points, n_points, -1)
    norm_vel = np.clip(np.linalg.norm(vel, axis=2), a_min=0, a_max=50)
    cmap = plt.cm.Greys
    color = 'black'
    plt.streamplot(
        x1_coords, x2_coords,
        vel[:, :, 0], vel[:, :, 1],
        color=color, cmap=cmap, linewidth=0.5,
        density=2, arrowstyle='fancy', arrowsize=1, zorder=2
    )

    CS = plt.contourf(x1_coords, x2_coords, norm_vel, cmap='viridis', levels=50, zorder=1)

    min_vel_ceil = np.ceil(np.min(norm_vel))
    max_vel_floor = np.ceil(np.max(norm_vel))
    delta_x = max_vel_floor / 10
    cbar = plt.colorbar(CS, location='bottom', ticks=np.arange(min_vel_ceil, max_vel_floor, delta_x))
    cbar.ax.set_xlabel('speed (mm/s)')

    plt.scatter(x_sim[-1, 0, 0], x_sim[-1, 1, 0], linewidth=4, color='blue', zorder=13)

    plt.xlim(x_lim[0])
    plt.ylim(x_lim[1])
    plt.tight_layout()
    plt.savefig('%s_vector_field.pdf' % data_name, dpi=300)
    plt.show()


if __name__ == '__main__':
        global options
        options['disp'] = 0

        options.update()

        main(Vxf0, options)
