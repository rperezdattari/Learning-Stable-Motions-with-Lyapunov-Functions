__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

import numpy as np

general_params = {
    'shape_name': 'PShape'
}

lyapunov_learner_params = {
    'L': 2,  # number of lyapunov function components
    'd': 2,
    'w': 1e-4,  # a positive scalar weight regulating the priority between the two objectives of the opitmization. Please refer to the page 7 of the paper for further information.
    'Mu': np.array(()),
    'P': np.array(()),
    'tol_mat_bias': 1e-1,
    'int_lyap_random': False,
    'optimizePriors': True,
    'upperBoundEigenValue': True
}

gmm_params = {
    'num_clusters': 10,
    'max_iterations': 100
}
