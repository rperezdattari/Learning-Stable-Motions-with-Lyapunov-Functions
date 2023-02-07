__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

import os
import scipy.io as sio
from stabilizer.ds_stab import dsStabilizer
import numpy as np


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