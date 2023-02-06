from __future__ import print_function

__author__ 		= "Olalekan Ogunmolu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Jethro Tan (PFN)"
__license__ 	= "MIT"
__maintainer__ 	= "Olalekan Ogunmolu"
__email__ 		= "patlekano@gmail.com"
__status__ 		= "Testing"

import os
import sys
# import argparse
# import logging
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from os.path import dirname, abspath
lyap = dirname(dirname(abspath(__file__)))
sys.path.append(lyap)

from cost.cost import Cost
from config import Vxf0, options
from utils.utils import guess_init_lyap
from stabilizer.ds_stab import dsStabilizer
from gmm.gmm import GMM

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# logger = logging.getLogger(__name__)
#
# parser = argparse.ArgumentParser(description='torobo_parser')
# parser.add_argument('--silent', '-si', type=int, default=0, help='max num iterations' )
# parser.add_argument('--data_type', '-dt', type=str, default='pipe_et_trumpet', help='pipe_et_trumpet | h5_data' )
# args = parser.parse_args()
#
# print(args)


def load_gmm_parameters(x):
    matFile = sio.loadmat(x)

    Priors_EM, Mu_EM, Sigma_EM = matFile['Priors_EM'], matFile['Mu_EM'], matFile['Sigma_EM']
    return Priors_EM, Mu_EM, Sigma_EM


def load_saved_mat_file(data_name):
    """
        Loads a matlab file from a subdirectory.

        Inputs:
            x: path to data on HDD


       Copyright (c) Lekan Molux. https://scriptedonachip.com
       2021.
    """

    # matFile = sio.loadmat(x)

    # data = matFile['Data']
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
    modelNames = ['w.mat', 'Sshape.mat']  # Two example models provided by Khansari
    modelNumber = 0  # could be zero or one depending on the experiment the user is running
    data_name = 'Sshape'
    data, demoIdx = load_saved_mat_file('Sshape')

    Vxf0['d'] = int(data.shape[0]/2)
    Vxf0.update(Vxf0)

    Vxf0 = guess_init_lyap(data, Vxf0, options['int_lyap_random'])
    cost = Cost()

    # cost.success = False
    while cost.success:
        # cost.success = False
        print('Optimizing the lyapunov function')
        Vxf, J = cost.learnEnergy(Vxf0, data, options)
        # if not cost.success:
        # increase L and restart the optimization process
        old_l = Vxf0['L']
        Vxf0['L'] += 1
        print('Constraints violated. increasing the size of L from {} --> {}'.format(old_l, Vxf0['L']))
        if cost.success:
            print('optimization succeeded without violating constraints')
            break

    # Plot the result of V
    #h1 = plt.plot(data[0, :], data[1, :], 'r.', label='demonstrations')

    extra = 30

    axes_limits = [np.min(data[0, :]) - extra, np.max(data[0, :]) + extra,
                   np.min(data[1, :]) - extra, np.max(data[1, :]) + extra]

    h3 = cost.energyContour(Vxf, axes_limits, np.array(()), np.array(()), np.array(()), False)
    h2 = plt.plot(0, 0, 'g*', markersize=15, linewidth=3, label='target')
    plt.title('Energy Levels of the learned Lyapunov Functions', fontsize=12)
    plt.xlabel('x (mm)', fontsize=15)
    plt.ylabel('y (mm)', fontsize=15)
    #h = [h1, h2, h3]

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

    #Priors_EM, Mu_EM, Sigma_EM = load_gmm_parameters(lyap + '/' + 'example_models/' + modelNames[modelNumber])

    # rho0 and kappa0 impose minimum acceptable rate of decrease in the energy
    # function during the motion. Refer to page 8 of the paper for more information
    rho0 = 1
    kappa0 = 0.1

    inp = list(range(Vxf['d']))
    output = np.arange(Vxf['d'], 2 * Vxf['d'])

    xd, _ = dsStabilizer(x0_all, Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost)

    # Evalute DS
    xT = np.array([])
    d = x0_all.shape[0]  # dimension of the model
    if not xT:
        xT = np.zeros((d, 1))

    # Simulate trajectories
    nbSPoint = x0_all.shape[1]
    x = []
    x.append(x0_all)
    xd = []
    if xT.shape == x0_all.shape:
        XT = xT
    else:
        XT = np.tile(xT, [1, nbSPoint])   # a matrix of target location (just to simplify computation)

    t = 0  # starting time
    dt = 0.01
    for i in range(4000):
        xd.append(dsStabilizer(x[i] - XT, Vxf, rho0, kappa0, priors, mu, sigma, inp, output, cost)[0])

        x.append(x[i] + xd[i] * dt)
        t += dt

    # Plot simulated data
    x = np.array(x)
    plt.plot(x[:, 0, :], x[:, 1, :], color='red', linewidth=4, zorder=10)


    # Plot demonstrations
    demo_length = 1000
    for i in range(int(data.shape[1] / demo_length)):
        plt.plot(data[0, i * demo_length:(i + 1) * demo_length], data[1, i * demo_length:(i + 1) * demo_length],
                 color='blue', linewidth=4, zorder=5)

    plt.legend()
    plt.savefig('%s_vector_field.pdf' % data_name, dpi=300)
    plt.show()


if __name__ == '__main__':
        global options
        # options = BundleType(options)
        # A set of options that will be passed to the solver
        options['disp'] = 0
        #options['args'] = args

        options.update()

        main(Vxf0, options)
