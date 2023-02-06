from __future__ import print_function

__author__ 		= "Olalekan Ogunmolu"
__copyright__ 	= "2018, One Hell of a Lyapunov Solver"
__credits__  	= "Rachel Thomson (MIT), Jethro Tan (PFN)"
__license__ 	= "MIT"
__maintainer__ 	= "Olalekan Ogunmolu"
__email__ 		= "patlekano@gmail.com"
__status__ 		= "Testing"

import logging, time
import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.linalg as LA
from scipy.optimize import minimize, linprog, NonlinearConstraint, BFGS
from gmm import gmm_2_parameters, parameters_2_gmm, \
                shape_DS, gmr_lyapunov
import matplotlib.pyplot as plt
LOGGER = logging.getLogger(__name__)


class Cost(object):

    def __init__(self):
        """
            Class that estimates lyapunov energy function
        """
        self.Nfeval = 0
        self.success = True   # boolean indicating if constraints were violated

    def matVecNorm(self, x):
        return np.sqrt(np.sum(x**2, axis=0))

    def obj(self, p, x, xd, d, L, w, options):
        if L == -1:  # SOS
            Vxf['n'] = np.sqrt(len(p) / d ** 2)
            Vxf['d'] = d
            Vxf['P'] = p.reshape(Vxf['n'] * d, Vxf['n'] * d)
            Vxf['SOS'] = 1
        else:
            Vxf = shape_DS(p, d, L, options)
            Vxf.update(Vxf)
        _, Vx = self.computeEnergy(x, np.array(()), Vxf, nargout=2)
        Vdot = np.sum(Vx * xd, axis=0)  # derivative of J w.r.t. xd
        norm_Vx = np.sqrt(np.sum(Vx * Vx, axis=0))
        norm_xd = np.sqrt(np.sum(xd * xd, axis=0))
        Vdot = np.expand_dims(Vdot, axis=0)
        norm_Vx = np.expand_dims(norm_Vx, axis=0)
        norm_xd = np.expand_dims(norm_xd, axis=0)
        butt = norm_Vx * norm_xd

        # w was added by Lekan to regularize the invalid values in butt
        J = Vdot / (butt + w)
        J[np.where(norm_xd == 0)] = 0
        J[np.where(norm_Vx == 0)] = 0
        J[np.where(Vdot > 0)] = J[np.where(Vdot > 0)] ** 2
        J[np.where(Vdot < 0)] = -w * J[np.where(Vdot < 0)] ** 2
        J = np.sum(J, axis=1)
        #print('J:', J[0])
        return J

    def callback_opt(self, Xi, y):
        print('Iteration: {0:4d}   Cost: {1: 3.6f}'.format(self.Nfeval, y.fun[0]))
        self.Nfeval += 1

    def optimize(self, obj_handle, ctr_handle_ineq, ctr_handle_eq, p0):
        nonl_cons_ineq = NonlinearConstraint(ctr_handle_ineq, -np.inf, 0, jac='3-point', hess=BFGS())
        nonl_cons_eq = NonlinearConstraint(ctr_handle_eq, 0, 0, jac='3-point', hess=BFGS())

        solution = minimize(obj_handle,
                            np.reshape(p0, [len(p0)]),
                            hess=BFGS(),
                            constraints=[nonl_cons_eq, nonl_cons_ineq],
                            method='trust-constr', options={'disp': True, 'initial_constr_penalty': 1.5},
                            callback=self.callback_opt)

        return solution.x, solution.fun

    def ctr_eigenvalue_ineq(self, p, d, L, options):
        # This function computes the derivative of the constrains w.r.t.
        # optimization parameters.
        Vxf = dict()
        if L == -1:  # SOS
            Vxf['d'] = d
            Vxf['n'] = int(np.sqrt(len(p) / d ** 2))
            Vxf['P'] = p.reshape(Vxf['n'] * d, Vxf['n'] * d)
            Vxf['SOS'] = 1
            c = np.zeros((Vxf['n'] * d, 1))
        else:
            Vxf = shape_DS(p, d, L, options)
            if L > 0:
                c = np.zeros(((L + 1) * d + (L + 1) * options['optimizePriors'], 1))  # +options.variableSwitch
            else:
                c = np.zeros((d, 1))

        if L == -1:  # SOS
            c = -np.linalg.eigvals(Vxf['P'] + Vxf['P'].T - np.eye(Vxf['n'] * d) * options['tol_mat_bias'])
        else:
            for k in range(L + 1):
                lambder = sp.linalg.eigvals(Vxf['P'][k, :, :] + (Vxf['P'][k, :, :]).T)
                lambder = np.divide(lambder.real, 2.0)
                lambder = np.expand_dims(lambder, axis=1)
                c[k * d:(k + 1) * d] = -lambder.real + options['tol_mat_bias']

        if L > 0 and options['optimizePriors']:
            c[(L + 1) * d:(L + 1) * d + L + 1] = np.reshape(-Vxf['Priors'], [L + 1, 1])

        return np.reshape(c, [len(c)])

    def ctr_eigenvalue_eq(self, p, d, L, options):
        # This function computes the derivative of the constrains w.r.t.
        # optimization parameters.
        Vxf = dict()
        if L == -1:  # SOS
            Vxf['d'] = d
            Vxf['n'] = int(np.sqrt(len(p) / d ** 2))
            Vxf['P'] = p.reshape(Vxf['n'] * d, Vxf['n'] * d)
            Vxf['SOS'] = 1
            ceq = np.array(())
        else:
            Vxf = shape_DS(p, d, L, options)
            if L > 0:
                if options['upperBoundEigenValue']:
                    ceq = np.zeros((L + 1, 1))
                else:
                    ceq = np.array(())  # zeros(L+1,1);
            else:
                ceq = (np.ravel(Vxf['P']).T).dot(np.ravel(Vxf['P'])) - 2

        if L == -1:  # SOS
            pass
        else:
            for k in range(L + 1):
                lambder = sp.linalg.eigvals(Vxf['P'][k, :, :] + (Vxf['P'][k, :, :]).T)
                lambder = np.divide(lambder.real, 2.0)
                lambder = np.expand_dims(lambder, axis=1)
                if options['upperBoundEigenValue']:
                    ceq[k] = 1.0 - np.sum(lambder.real)  # + Vxf.P(:,:,k+1)'

        return np.reshape(ceq, [len(ceq)])

    def check_constraints(self, p, ctr_handle, d, L, options):
        c = -ctr_handle(p)

        if L > 0:
            c_P = c[:L*d].reshape(d, L).T
        else:
            c_P = c

        i = np.where(c_P <= 0)
        # self.success = True

        if i:
            self.success = False
        else:
            self.success = True

        if L > 1:
            if options['optimizePriors']:
                c_Priors = c[L*d+1:L*d+L]
                i = np.nonzero(c_Priors < 0)

                if i:
                    self.success = False
                else:
                    self.success = True

            if len(c) > L*d+L:
                c_x_sw = c[L*d+L+1]
                if c_x_sw <= 0:
                    self.success = False
                else:
                    self.success = True


    def computeEnergy(self, X, Xd, Vxf, nargout=2):
        d = X.shape[0]
        nDemo = 1
        if nDemo>1:
            X = X.reshape(d,-1)
            Xd = Xd.reshape(d,-1)

        if Vxf['SOS']:
            V, dV = sos_lyapunov(X, Vxf['P'], Vxf['d'], Vxf['n'])
            if 'p0' in Vxf:
                V -= Vxf['p0']
        else:
            V, dV = gmr_lyapunov(X, Vxf['Priors'], Vxf['Mu'], Vxf['P'])

        if nargout > 1:
            if not Xd:
                Vdot = dV
            else:
                Vdot = np.sum(Xd*dV, axis=0)
        if nDemo>1:
            V = V.reshape(-1, nDemo).T
            if nargout > 1:
                Vdot = Vdot.reshape(-1, nDemo).T

        return V, Vdot

    def learnEnergy(self, Vxf0, Data, options):
        d = Vxf0['d']
        x = Data[:d, :]
        xd = Data[d:, :]

        # Transform the Lyapunov model to a vector of optimization parameters
        if Vxf0['SOS']:
            p0 = npr.randn(d * Vxf0['n'], d * Vxf0['n'])
            p0 = p0.dot(p0.T)
            p0 = np.ravel(p0)
            Vxf0['L'] = -1  # to distinguish sos from other methods
        else:
            for l in range(Vxf0['L']):
                try:
                    Vxf0['P'][l + 1, :, :] = sp.linalg.solve(Vxf0['P'][l + 1, :, :], sp.eye(d))
                except sp.linalg.LinAlgError as e:
                    LOGGER.debug('LinAlgError: %s', e)

            # in order to set the first component to be the closest Gaussian to origin
            to_sort = self.matVecNorm(Vxf0['Mu'])
            idx = np.argsort(to_sort, kind='mergesort')
            Vxf0['Mu'] = Vxf0['Mu'][:, idx]
            Vxf0['P'] = Vxf0['P'][idx, :, :]
            p0 = gmm_2_parameters(Vxf0, options)

        # account for targets in x and xd
        # x = x - np.expand_dims(x[:, -1], 1)
        # xd = xd - np.expand_dims(xd[:, -1], 1)

        obj_handle = lambda p: self.obj(p, x, xd, d, Vxf0['L'], Vxf0['w'], options)
        ctr_handle_ineq = lambda p: self.ctr_eigenvalue_ineq(p, d, Vxf0['L'], options)
        ctr_handle_eq = lambda p: self.ctr_eigenvalue_eq(p, d, Vxf0['L'], options)

        popt, J = self.optimize(obj_handle, ctr_handle_ineq, ctr_handle_eq, p0)
        # popt, J = optim_res.x, optim_res.fun

        #while not optim_res.success:
         #   optim_res   = self.optimize(obj_handle, ctr_handle, p0)
          #  popt, J     = optim_res.x, optim_res.fun

        if Vxf0['SOS']:
            Vxf['d']    = d
            Vxf['n']    = Vxf0['n']
            Vxf['P']    = popt.reshape(Vxf['n']*d,Vxf['n']*d)
            Vxf['SOS']  = 1
            Vxf['p0']   = self.computeEnergy(zeros(d,1),[],Vxf)
            self.check_constraints(popt,ctr_handle,d,0,options)
        else:
            # transforming back the optimization parameters into the GMM model
            Vxf             = parameters_2_gmm(popt,d,Vxf0['L'],options)
            Vxf['Mu'][:,0]  = 0
            Vxf['L']        = Vxf0['L']
            Vxf['d']        = Vxf0['d']
            Vxf['w']        = Vxf0['w']
#            self.check_constraints(popt,ctr_handle,d,Vxf['L'],options) TODO: check check_constraints method
        self.success = True

        sumDet = 0
        for l in range(Vxf['L'] + 1):
            sumDet += np.linalg.det(Vxf['P'][l, :, :])

        Vxf['P'][0, :, :] = Vxf['P'][0, :, :] / sumDet
        Vxf['P'][1:, :, :] = Vxf['P'][1:, :, :] / np.sqrt(sumDet)

        return Vxf, J

    def energyContour(self, Vxf, D, *args):
        quality='low'
        b_plot_contour = True
        contour_levels = np.array([])

        if quality == 'high':
            nx, ny = 0.1, 0.1
        elif quality == 'medium':
            nx, ny = 1, 1
        else:
            nx, ny = 2, 2

        x = np.arange(D[0], D[1], nx)
        y = np.arange(D[2], D[3], ny)
        x_len = len(x)
        y_len = len(y)
        X, Y = np.meshgrid(x, y)
        x = np.stack([np.ravel(X), np.ravel(Y)])

        V, dV = self.computeEnergy(x, np.array(()), Vxf, nargout=2)

        if not contour_levels.size:
            contour_levels = np.arange(0, np.log(np.max(V)), 0.5)
            contour_levels = np.exp(contour_levels)
            if np.max(V) > 40:
                contour_levels = np.round(contour_levels)

        V = V.reshape(y_len, x_len)

        if b_plot_contour:
            h = plt.contour(X, Y, V, contour_levels, colors='k', origin='upper', linewidths=2, labelspacing=200)

        return h
