__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.linalg as LA
from scipy.optimize import minimize, NonlinearConstraint, BFGS
from gmm import gmm_2_parameters, parameters_2_gmm, shape_DS, gmr_lyapunov
import matplotlib.pyplot as plt


class LyapunovLearner():
    def __init__(self):
        """
            Class that estimates lyapunov energy function
        """
        self.Nfeval = 0
        self.success = True   # boolean indicating if constraints were violated

    def guess_init_lyap(self, Vxf0):
        """
        This function guesses the initial lyapunov function
        """
        b_initRandom = Vxf0['int_lyap_random']
        Vxf0['Mu'] = np.zeros((Vxf0['d'], Vxf0['L'] + 1))
        Vxf0['P'] = np.zeros((Vxf0['d'], Vxf0['d'], Vxf0['L'] + 1))

        if b_initRandom:
            """
             If `rowvar` is True (default), then each row represents a
            variable, with observations in the columns. Otherwise, the relationship
            is transposed: each column represents a variable, while the rows
            contain observations.
            """
            Vxf0['Priors'] = np.random.rand(Vxf0['L'] + 1, 1)

            for l in range(Vxf0['L'] + 1):
                tempMat = np.random.randn(Vxf0['d'], Vxf0['d'])
                Vxf0['Mu'][:, l] = np.random.randn(Vxf0['d'])
                Vxf0['P'][:, :, l] = tempMat
        else:
            Vxf0['Priors'] = np.ones((Vxf0['L'] + 1, 1))
            Vxf0['Priors'] = Vxf0['Priors'] / np.sum(Vxf0['Priors'])
            Vxf0['P'] = []
            for l in range(Vxf0['L'] + 1):
                Vxf0['P'].append(np.eye(Vxf0['d'], Vxf0['d']))

            Vxf0['P'] = np.reshape(Vxf0['P'], [Vxf0['L'] + 1, Vxf0['d'], Vxf0['d']])

        Vxf0.update(Vxf0)

        return Vxf0

    def matVecNorm(self, x):
        return np.sqrt(np.sum(x**2, axis=0))

    def obj(self, p, x, xd, d, L, w, options):
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

        J = Vdot / (butt + w)
        J[np.where(norm_xd == 0)] = 0
        J[np.where(norm_Vx == 0)] = 0
        J[np.where(Vdot > 0)] = J[np.where(Vdot > 0)] ** 2
        J[np.where(Vdot < 0)] = -w * J[np.where(Vdot < 0)] ** 2
        J = np.sum(J, axis=1)
        return J

    def callback_opt(self, Xi, y):
        print('Iteration: {0:4d}   Cost: {1: 3.6f}'.format(self.Nfeval, y.fun))
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
        Vxf = shape_DS(p, d, L, options)
        if L > 0:
            c = np.zeros(((L + 1) * d + (L + 1) * options['optimizePriors'], 1))  # +options.variableSwitch
        else:
            c = np.zeros((d, 1))

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
        Vxf = shape_DS(p, d, L, options)
        if L > 0:
            if options['upperBoundEigenValue']:
                ceq = np.zeros((L + 1, 1))
            else:
                ceq = np.array(())  # zeros(L+1,1);
        else:
            ceq = (np.ravel(Vxf['P']).T).dot(np.ravel(Vxf['P'])) - 2

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
        for l in range(Vxf0['L']):
            try:
                Vxf0['P'][l + 1, :, :] = sp.linalg.solve(Vxf0['P'][l + 1, :, :], sp.eye(d))
            except sp.linalg.LinAlgError as e:
                print('Error lyapunov solver.')

        # in order to set the first component to be the closest Gaussian to origin
        to_sort = self.matVecNorm(Vxf0['Mu'])
        idx = np.argsort(to_sort, kind='mergesort')
        Vxf0['Mu'] = Vxf0['Mu'][:, idx]
        Vxf0['P'] = Vxf0['P'][idx, :, :]
        p0 = gmm_2_parameters(Vxf0, options)

        # account for targets in x and xd
        obj_handle = lambda p: self.obj(p, x, xd, d, Vxf0['L'], Vxf0['w'], options)
        ctr_handle_ineq = lambda p: self.ctr_eigenvalue_ineq(p, d, Vxf0['L'], options)
        ctr_handle_eq = lambda p: self.ctr_eigenvalue_eq(p, d, Vxf0['L'], options)

        popt, J = self.optimize(obj_handle, ctr_handle_ineq, ctr_handle_eq, p0)

        # transforming back the optimization parameters into the GMM model
        Vxf = parameters_2_gmm(popt,d,Vxf0['L'],options)
        Vxf['Mu'][:, 0] = 0
        Vxf['L'] = Vxf0['L']
        Vxf['d'] = Vxf0['d']
        Vxf['w'] = Vxf0['w']
        self.success = True

        sumDet = 0
        for l in range(Vxf['L'] + 1):
            sumDet += np.linalg.det(Vxf['P'][l, :, :])

        Vxf['P'][0, :, :] = Vxf['P'][0, :, :] / sumDet
        Vxf['P'][1:, :, :] = Vxf['P'][1:, :, :] / np.sqrt(sumDet)

        return Vxf, J

    def energyContour(self, Vxf, D):
        quality ='high'
        b_plot_contour = True
        contour_levels = np.array([])

        if quality == 'high':
            nx, ny = 0.1, 0.1
        elif quality == 'medium':
            nx, ny = 1, 1
        else:
            nx, ny = 2, 2

        x = np.arange(D[0][0], D[0][1], nx)
        y = np.arange(D[1][0], D[1][1], ny)
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
