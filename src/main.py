__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

from lyapunov_learner.lyapunov_learner import LyapunovLearner
from config import gmm_params, lyapunov_learner_params, general_params
from stabilizer.ds_stab import DynamicalSystem
from gmm.gmm import GMM
from tools import load_saved_mat_file, plot_results


def main(data_name):
    # Load demonstrations
    data, demo_idx, demo_length = load_saved_mat_file(data_name)

    # Initialize Lyapunov learner
    lyapunov = LyapunovLearner()
    V_init = lyapunov.guess_init_lyap(lyapunov_learner_params)
    V = None

    # Optimize Lyapunov function
    while lyapunov.success:
        print('Optimizing the lyapunov function')
        V, J = lyapunov.learnEnergy(V_init, data, lyapunov_learner_params)
        if lyapunov.success:
            print('optimization succeeded without violating constraints')
            break

    # Optimize GMM
    gmm = GMM(num_clusters=gmm_params['num_clusters'])
    gmm.update(data.T, K=gmm_params['num_clusters'], max_iterations=gmm_params['max_iterations'])
    mu, sigma, priors = gmm.mu.T, gmm.sigma.T, gmm.logmass.T

    # Create dynamical system from learned GMM and Lyapunov corrections
    dynamical_system = DynamicalSystem(V, priors, mu, sigma, lyapunov)

    # Plot results
    plot_results(data, data_name, demo_idx, demo_length, dynamical_system, plot_mode='velocities')
    plot_results(data, data_name, demo_idx, demo_length, dynamical_system, plot_mode='energy_levels')


if __name__ == '__main__':
    data_name = general_params['shape_name']
    main(data_name)
