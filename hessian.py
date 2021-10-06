from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from jax import grad, jit, jacfwd, jacrev
import jax.numpy as jnp


class HessianCircadian:

    def __init__(self):
        self._default_params()


    def hessian(self, u0: jnp.ndarray = jnp.array([0.70,0.0,0.0]), light: jnp.ndarray = jnp.zeros(240*5)):
        params = self.get_parameters_array()
        statefinal = self.step_n(u0, light, params, 0.10)
        def loss(params): return HessianCircadian.norm(HessianCircadian.step_n(u0, light, params, 0.10), statefinal)
        H=jacfwd(jacrev(loss))
        return H(params)

    def _default_params(self):
        """
            Load the model parameters, if useFile is False this will search the local directory for a optimalParams.dat file.

            setParameters()

            No return value
        """
        default_params = {'tau': 23.84, 'K': 0.06358, 'gamma': 0.024,
                          'Beta1': -0.09318, 'A1': 0.3855, 'A2': 0.1977,
                          'BetaL1': -0.0026, 'BetaL2': -0.957756, 'sigma': 0.0400692,
                          'G': 33.75, 'alpha_0': 0.05, 'delta': 0.0075,
                          'p': 1.5, 'I0': 9325.0}

        self.set_parameters(default_params)

    def get_parameters_array(self):
        """
            Return a numpy array of the models current parameters
        """
        return jnp.array([self.tau, self.K, self.gamma, self.Beta1, self.A1, self.A2, self.BetaL1, self.BetaL2, self.sigma, self.G, self.alpha_0, self.delta, self.I0, self.p])

    def set_parameters(self, param_dict: dict):
        """
            Update the model parameters using a passed in parameter dictionary. Any parameters not included
            in the dictionary will be set to the default values.

            updateParameters(param_dict)

            Returns null, changes the parameters stored in the class instance
        """

        params = ['tau', 'K', 'gamma', 'Beta1', 'A1', 'A2', 'BetaL1',
                  'BetaL2', 'sigma', 'G', 'alpha_0', 'delta', 'p', 'I0']

        for key, value in param_dict.items():
            setattr(self, key, value)

    @staticmethod
    @jit
    def spmodel(u: jnp.array, light: float, params: jnp.array):

        R, Psi, n = u

        tau, K, gamma, Beta1, A1, A2, BetaL1, BetaL2, sigma, G, alpha_0, delta, I0, p = params
            

        alpha_0_func = alpha_0 * pow(light, p) / (pow(light, p) + I0)
        Bhat = G * (1.0 - n) * alpha_0_func
        LightAmp = A1 * 0.5 * Bhat * (1.0 - pow(R, 4.0)) * jnp.cos(Psi + BetaL1) + \
                   A2 * 0.5 * Bhat * R * (1.0 - pow(R, 8.0)) * jnp.cos(2.0 * Psi + BetaL2)
        LightPhase = sigma * Bhat - A1 * Bhat * 0.5 * (pow(R, 3.0) + 1.0 / R) * jnp.sin(
            Psi + BetaL1) - A2 * Bhat * 0.5 * (1.0 + pow(R, 8.0)) * jnp.sin(2.0 * Psi + BetaL2)

        dR = -1.0 * gamma * R + K * jnp.cos(Beta1) / 2.0 * R * (1.0 - pow(R, 4.0)) + LightAmp
        dPsi = 2 * jnp.pi / tau + K / 2.0 * jnp.sin(Beta1) * (1 + pow(R, 4.0)) + LightPhase
        dn = 60.0 * (alpha_0_func * (1.0 - n) - delta * n)

        du = jnp.array([dR, dPsi, dn])
        return (du)

    @staticmethod
    def spmodel_rk4_step(ustart: jnp.ndarray, light_val: float, params: jnp.array, dt: float):
        """
            Takes a single step forward using the default spmodel
        """

        k1 = HessianCircadian.spmodel(ustart, light_val, params)
        k2 = HessianCircadian.spmodel(ustart + dt / 2 * k1, light_val, params)
        k3 = HessianCircadian.spmodel(ustart + dt / 2 * k2, light_val, params)
        k4 = HessianCircadian.spmodel(ustart + dt * k3, light_val, params)
        return ustart + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    @staticmethod
    def step_n(u0: jnp.ndarray, light: jnp.ndarray, params: jnp.ndarray, dt: float):
        for k in range(len(light)):
            u0 = HessianCircadian.spmodel_rk4_step(u0, light[k], params, dt)
        return u0

    @staticmethod
    @jit
    def norm(s1: jnp.ndarray, s2: jnp.ndarray):
        x1 = s1[0] * jnp.cos(s1[1])
        y1 = s1[0] * jnp.sin(s1[1])
        x2 = s2[0] * jnp.cos(s2[1])
        y2 = s2[0] * jnp.sin(s2[1])

        return (x1 - x2) ** 2 + (y1 - y2) ** 2



if __name__ == '__main__':

    from lightschedules import RegularLight
    ts=np.arange(0, 24*5, 0.10)
    lights = jnp.array([ RegularLight(t) for t in ts ])
    lights=np.zeros(len(ts))
    sens=HessianCircadian()

    hessianVal = sens.hessian(light=lights)

    hessianEigs=np.linalg.eigvals(hessianVal)
    print(hessianEigs)
    plt.scatter(range(len(hessianEigs)), hessianEigs)
    plt.show()

    print(hessianVal)