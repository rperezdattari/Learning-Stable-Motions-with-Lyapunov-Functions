__credits__ = 'Olalekan Ogunmolu, Rodrigo Perez-Dattari (TU Delft), Rachel Thomson (MIT), Jethro Tan (PFN)'
__license__ = 'MIT'

import numpy as np


class DynamicalSystem():
    def __init__(self, Vxf, priors, mu, sigma, lyapunov):
        """
            Class that contains the learned dynamical system + corrections
        """
        self.Vxf = Vxf
        self.priors = priors
        self.mu = mu
        self.sigma = sigma
        self.lyapunov = lyapunov

        self.rho0 = 1  # rho0 and kappa0 impose minimum acceptable rate of decrease in the energy function during the motion
        self.kappa0 = 0.1  # refer to page 8 of the paper for more information
        self.input = np.arange(0, Vxf['d'])
        self.output = np.arange(Vxf['d'], 2 * Vxf['d'])

    def ds_stabilizer(self, x):
        d = self.Vxf['d']
        if x.shape[0] == 2*d:
            xd = x[d+1:2*d, :]
            x = x[:d, :]
        else:
            xd, _, _ = self.GMR(x)
        V, Vx = self.lyapunov.computeEnergy(x, np.array(()), self.Vxf)
        norm_Vx = np.sum(Vx * Vx, axis=0)
        norm_x = np.sum(x * x, axis=0)
        Vdot = np.sum(Vx * xd, axis=0)
        rho = self.rho0 * (1-np.exp(-self.kappa0 * norm_x)) * np.sqrt(norm_Vx)
        ind = Vdot + rho >= 0
        u = xd * 0

        if np.sum(ind) > 0:
            lambder = (Vdot[ind] + rho[ind]) / (norm_Vx[ind] + 1e-8)
            u[:, ind] = -np.tile(lambder, [d, 1]) * Vx[:, ind]
            xd[:, ind] = xd[:, ind] + u[:, ind]

        return xd, u

    def GMR(self, x, nargout=0):
        nbData = x.shape[1]
        nbStates = self.sigma.shape[2]

        ## Fast matrix computation (see the commented code for a version involving
        ## one-by-one computation, which is easier to understand).
        ##
        ## Compute the influence of each GMM component, given input x
        #########################################################################
        Pxi = []
        for i in range(nbStates):
            Pxi.append(self.priors[0, i] * self.gaussPDF(x, self.mu[self.input, i], self.sigma[self.input[0]:(self.input[1] + 1), self.input[0]:(self.input[1] + 1), i]))

        Pxi = np.reshape(Pxi, [len(Pxi), -1]).T
        beta = Pxi / np.tile(np.sum(Pxi, axis=1) + 1e-300, [nbStates, 1]).T

        #########################################################################
        y_tmp = []
        for j in range(nbStates):
            a = np.tile(self.mu[self.output, j], [nbData, 1]).T
            b = self.sigma[self.output, self.input[0]:(self.input[1] + 1), j]
            c = x - np.tile(self.mu[self.input[0]:(self.input[1] + 1), j], [nbData, 1]).T
            d = self.sigma[self.input[0]:(self.input[1] + 1), self.input[0]:(self.input[1] + 1), j]
            e = np.linalg.lstsq(d, b.T)[0].T
            y_tmp.append(a + e.dot(c))

        y_tmp = np.reshape(y_tmp, [nbStates, len(self.output), nbData])

        beta_tmp = beta.T.reshape([beta.shape[1], 1, beta.shape[0]])
        y_tmp2 = np.tile(beta_tmp, [1, len(self.output), 1]) * y_tmp
        y = np.sum(y_tmp2, axis=0)
        ## Compute expected covariance matrices Sigma_y, given input x
        #########################################################################
        Sigma_y_tmp = []
        Sigma_y = []
        if nargout > 1:
            for j in range(nbStates):
                Sigma_y_tmp.append(
                    self.sigma[self.output, self.output, j] - (self.sigma[self.output, self.input, j] / (self.sigma[self.input, self.input, j]) * self.sigma[self.input, self.output, j]))

            beta_tmp = beta.reshape(1, 1, beta.shape)
            Sigma_y_tmp2 = np.tile(beta_tmp * beta_tmp, [len(self.output), len(self.output), 1, 1]) * np.tile(Sigma_y_tmp,
                                                                                              [1, 1, nbData, 1])
            Sigma_y = np.sum(Sigma_y_tmp2, axis=3)
        return y, Sigma_y, beta

    def gaussPDF(self, data, mu, sigma):
        nbVar, nbdata = data.shape

        data = data.T - np.tile(mu.T, [nbdata, 1])
        prob = np.sum(np.linalg.lstsq(sigma, data.T)[0].T * data, axis=1)
        prob = np.exp(-0.5 * prob) / np.sqrt((2 * np.pi) ** nbVar * np.abs(np.linalg.det(sigma) + 1e-300))

        return prob.T