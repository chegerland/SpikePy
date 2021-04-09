#!/usr/bin/env python3

from math import pi
import numpy as np
import mpmath as mp
from scipy import optimize
from scipy import integrate
from scipy import special

import configparser

mp.dps = 15
mp.pretty = True


def integrand(x):
    """
    Returns the integrand to calculate the firing rate of the LIF (e^x^2 * erfc(x))
    :param x: x
    :return: f(x) = e^(x^2)*erfc(x)
    """
    return np.exp(x * x) * special.erfc(x)


class CosineSignal:
    """
    A class for a cosine signals
    """

    def __init__(self, alpha, f):
        self.alpha = alpha
        self.f = f

    @classmethod
    def from_ini(cls, ini_path):
        """
        Initialize CosineSignal from .ini file
        :param ini_path: path to .ini
        :return: CosineSignal
        """
        config = configparser.ConfigParser()
        config.read(ini_path)

        neuron = config['Signal']
        assert (neuron['type'] == "cosine")

        alpha = float(neuron['alpha'])
        f = float(neuron['f'])

        return cls(alpha, f)

    def firing_rate_linear(self, neuron, t):
        """
        Returns the linear response of a neuron for a single cosine signal.
        :param neuron: the neuron
        :return: r(t) = r_0 + alpha*|chi_1(omega)| cos(omega t - phi)
        """
        omega = 2 * pi * self.f
        stat = neuron.stationary_rate()
        lr = self.alpha * np.fabs(neuron.susceptibility_1(omega)) * \
            np.cos(omega * t - np.arg(neuron.susceptibility_1(omega)))
        result = stat + lr
        #assert np.imag(result) == 0
        return np.real(result)

    def firing_rate_nonlinear(self, neuron, t):
        """
        Returns the nonlinear response of a neuron for a single cosine signal.
        :param neuron: the neuron
        :return: r(t) = r_0 + alpha*|chi_1(omega)| cos(omega t - phi) + higher harmonics
        """
        omega = 2 * pi * self.f
        stat = neuron.stationary_rate() + 0.5 * self.alpha ** 2 * \
            neuron.susceptibility_2(omega, -omega)
        lr = self.alpha * np.absolute(neuron.susceptibility_1(omega)) * np.cos(
            omega * t - np.angle(neuron.susceptibility_1(omega)))
        hh = 0.5 * self.alpha ** 2 * np.absolute(neuron.susceptibility_2(omega, omega)) * np.cos(
            2 * omega * t - np.angle(neuron.susceptibility_2(omega, omega)))
        result = stat + lr + hh
        #assert np.imag(result) == 0
        return np.real(result)


class TwoCosineSignal():
    """
    A class for a signal that is the sum of two cosine signals
    """

    def __init__(self, alpha, f1, beta, f2):
        self.alpha = alpha
        self.f1 = f1
        self.beta = beta
        self.f2 = f2

    @classmethod
    def from_ini(cls, ini_path):
        """
        Initialize TwoCosineSignal from .ini file
        :param ini_path: path to .ini
        :return: TwoCosineSignal
        """
        config = configparser.ConfigParser()
        config.read(ini_path)

        neuron = config['Signal']
        assert (neuron['type'] == "two cosine")

        alpha = float(neuron['alpha'])
        f1 = float(neuron['f1'])
        beta = float(neuron['beta'])
        f2 = float(neuron['f2'])

        return cls(alpha, f1, beta, f2)

    def firing_rate_linear(self, neuron, t):
        """
        Returns the linear response of a neuron for a two cosine signal.
        :param neuron: the neuron
        :return: r(t) = r_0 + alpha*|chi_1(omega_1)| cos(omega_1 t - phi_1) + beta*|chi_1(omega_2)| cos(omega_2 - phi_2)
        """
        omega1 = 2 * pi * self.f1
        omega2 = 2 * pi * self.f2
        stat = neuron.stationary_rate()
        lr = self.alpha * np.absolute(neuron.susceptibility_1(omega1)) * np.cos(
            omega1 * t - np.angle(neuron.susceptibility_1(omega1))) + self.beta * np.absolute(
            neuron.susceptibility_1(omega2)) * np.cos(omega2 * t - np.angle(neuron.susceptibility_1(omega2)))
        result = stat + lr
        #assert np.imag(result) == 0
        return np.real(result)

    def firing_rate_nonlinear(self, neuron, t):
        """
        Returns the nonlinear response of a neuron for a two cosine signal.
        :param neuron: the neuron
        :return: r(t) = r_0 + linear response + higher harmonics + mixed response
        """
        omega1 = 2 * pi * self.f1
        omega2 = 2 * pi * self.f2
        stat = neuron.stationary_rate() \
            + 0.5 * self.alpha ** 2 * neuron.susceptibility_2(omega1, -omega1) \
            + 0.5 * self.alpha ** 2 * neuron.susceptibility_2(omega1, -omega1)
        lr = self.alpha * np.absolute(neuron.susceptibility_1(omega1)) * np.cos(
            omega1 * t - np.angle(neuron.susceptibility_1(omega1))) + self.beta * np.absolute(
            neuron.susceptibility_1(omega2)) * np.cos(omega2 * t - np.angle(neuron.susceptibility_1(omega2)))
        hh = 0.5 * self.alpha ** 2 * np.absolute(neuron.susceptibility_2(omega1, omega1)) * np.cos(
            2 * omega1 * t - np.angle(neuron.susceptibility_2(omega1, omega1))) + 0.5 * self.beta ** 2 * np.absolute(
            neuron.susceptibility_2(omega2, omega2)) * np.cos(
            2 * omega2 * t - np.angle(neuron.susceptibility_2(omega2, omega2)))
        mr = self.alpha * self.beta * (np.absolute(neuron.susceptibility_2(omega1, omega2)) * np.cos(
            (omega1 + omega2) * t - np.angle(neuron.susceptibility_2(omega1, omega2))) + np.absolute(
            neuron.susceptibility_2(omega1, -omega2)) * np.cos(
            (omega1 - omega2) * t - np.angle(neuron.susceptibility_2(omega1, -omega2))))
        result = stat + lr + hh + mr
        #assert np.imag(result) == 0
        return np.real(result)


class PIF:
    """
    A class for the perfect integrate-and-fire neuron
    """

    def __init__(self, mu, d):
        """
        Initialize PIF from parameters
        :param mu: mean input current
        :param d: noise coefficient
        """
        self.mu = mu
        self.D = d

    @classmethod
    def from_ini(cls, ini_path):
        """
        Initialize PIF from .ini file
        :param ini_path: path to .ini
        :return: PIF
        """
        config = configparser.ConfigParser()
        config.read(ini_path)

        neuron = config['Neuron']
        assert (neuron['type'] == "PIF")

        mu = float(neuron['mu'])
        d = float(neuron['D'])

        return cls(mu, d)

    def stationary_rate(self):
        return self.mu

    def susceptibility_1(self, omega):
        tau_e = 2.0 * self.D / (self.mu * self.mu)
        return (1.0 - np.sqrt(1.0 - 2.0j * tau_e * omega)) / (1.0j * tau_e * omega)


class PIFAC:
    def __init__(self, mu, d, delta, tau_a):
        self.mu = mu
        self.D = d
        self.Delta = delta
        self.tau_a = tau_a
        self.pif = PIF(mu * 1.0 / (1.0 + delta * tau_a), d)

    @classmethod
    def from_ini(cls, ini_path):
        config = configparser.ConfigParser()
        config.read(ini_path)

        neuron = config['Neuron']
        assert (neuron['type'] == "PIFAC")

        mu = float(neuron['mu'])
        d = float(neuron['D'])
        delta = float(neuron['Delta'])
        tau_a = float(neuron['tau_a'])

        return cls(mu, d, delta, tau_a)

    def susceptibility_1(self, omega):
        result = self.pif.susceptibility_1(omega)
        result = np.real(np.cdouble(result)) + 1.0j * \
            np.imag(np.cdouble(result))
        result = result / (1.0 + result * self.Delta *
                           self.tau_a / (1.0 - 1.0j * self.tau_a * omega))
        return result


class LIF:
    def __init__(
            self,
            mu,  # mean input current
            d  # diffusion coefficient
    ):
        self.mu = mu
        self.D = d
        self.r0 = self.stationary_rate()

    @classmethod
    def from_ini(cls, ini_path):
        config = configparser.ConfigParser()
        config.read(ini_path)

        neuron = config['Neuron']
        assert (neuron['type'] == "LIF")

        mu = float(neuron['mu'])
        d = float(neuron['D'])

        return cls(mu, d)

    def stationary_rate(self):
        lower = (self.mu - 1.0) / np.sqrt(2.0 * self.D)
        upper = self.mu / np.sqrt(2.0 * self.D)
        res, err = integrate.quad(integrand, lower, upper, epsrel=1e-5)
        r0 = 1.0 / (np.sqrt(pi) * res)
        return r0

    def susceptibility_1(self, omega):
        if omega == 0:
            omega = 1e-7
        alpha = self.r0 / mp.sqrt(self.D) * 1.0j * omega / (1.0j * omega - 1.0)
        a = mp.pcfd(1.0j * omega - 1.0, (self.mu - 1.0) / mp.sqrt(self.D))
        b = mp.exp((2.0 * self.mu - 1.0) / (4.0 * self.D)) * \
            mp.pcfd(1.0j * omega - 1.0, self.mu / mp.sqrt(self.D))
        c = mp.pcfd(1.0j * omega, (self.mu - 1.0) / mp.sqrt(self.D))
        d = mp.exp((2.0 * self.mu - 1.0) / (4.0 * self.D)) * \
            mp.pcfd(1.0j * omega, self.mu / mp.sqrt(self.D))

        result = np.cdouble(alpha * (a - b) / (c - d))
        return result

    def susceptibility_2(self, omega_1, omega_2):
        mu = self.mu
        D = self.D
        r0 = self.r0
        chi_1 = self.susceptibility_1

        # check if we got the case omega_2 = -omega_1
        if omega_2 == -omega_1:
            alpha = r0 * (1.0j * omega_1 + 1.0j) / (
                2.0 * D * (1.0j * omega_1 - 1.0) * (-1.0j * omega_1 - 1.0) * -1.0j * mp.exp(-(mu - 1.0)**2 / (4*D)))
            a = mp.pcfd(-2.0, (mu - 1.0) / mp.sqrt(D))
            b = mp.exp((2.0 * mu - 1.0) / (4.0 * D)) * \
                mp.pcfd(-2.0, mu / mp.sqrt(D))
            c = (mu - 1.0) / (2.0 * mp.sqrt(D))
            d = mu / (2.0 * mp.sqrt(D))

            beta = (1.0j * omega_1 + 1.0j) / (2.0 * mp.sqrt(D)
                                              * -1.0j * mp.exp(-(mu - 1.0)**2 / (4*D)))
            a_2 = (chi_1(omega_1) / (-1.0j * omega_1 - 1.0) + chi_1(-omega_1) / (
                1.0j * omega_1 - 1.0)) * mp.pcfd(- 1.0, (mu - 1.0) / mp.sqrt(D))
            a_3 = (chi_1(omega_1) / (-1.0j * omega_1 - 1.0) + chi_1(-omega_1) / (
                1.0j * omega_1 - 1.0)) * mp.exp((2.0 * mu - 1.0) / (4.0 * D)) * mp.pcfd(-1.0, mu / mp.sqrt(D))
        else:
            alpha = r0 * (1.0 - 1.0j * omega_1 - 1.0j * omega_2) * (1.0j * omega_1 + 1.0j * omega_2) / (
                2.0 * D * (1.0j * omega_1 - 1.0) * (1.0j * omega_2 - 1.0))
            a = mp.pcfd(1.0j * omega_1 + 1.0j * omega_2 - 2.0,
                        (mu - 1.0) / mp.sqrt(D))
            b = mp.exp((2.0 * mu - 1.0) / (4.0 * D)) * mp.pcfd(1.0j * omega_1 + 1.0j * omega_2 - 2.0,
                                                               mu / mp.sqrt(D))
            c = mp.pcfd(1.0j * omega_1 + 1.0j * omega_2,
                        (mu - 1.0) / mp.sqrt(D))
            d = mp.exp((2.0 * mu - 1.0) / (4.0 * D)) * mp.pcfd(1.0j * omega_1 + 1.0j * omega_2,
                                                               mu / mp.sqrt(D))

            beta = (1.0j * omega_1 + 1.0j * omega_2) / (2.0 * mp.sqrt(D))
            a_2 = (chi_1(omega_1) / (1.0j * omega_2 - 1.0) + chi_1(omega_2) / (
                1.0j * omega_1 - 1.0)) * mp.pcfd(1.0j * omega_1 + 1.0j * omega_2 - 1.0,
                                                 (mu - 1.0) / mp.sqrt(D))
            a_3 = (chi_1(omega_1) / (1.0j * omega_2 - 1.0) + chi_1(omega_2) / (
                1.0j * omega_1 - 1.0)) * mp.exp((2 * mu - 1.0) / (4.0 * D)) * mp.pcfd(
                1.0j * omega_1 + 1.0j * omega_2 - 1.0, mu / mp.sqrt(D))

        result = np.cdouble(alpha * (a - b) / (c - d) +
                            beta * a_2 / (c - d) - beta * a_3 / (c - d))
        return result

    def power_spectrum(self, omega):
        mu_vt = (self.mu - 1.0) / (mp.sqrt(self.D))
        mu_vr = self.mu / (mp.sqrt(self.D))
        Delta = (2.0 * self.mu - 1.0) / (4.0 * self.D)
        a = mp.pcfd(1.0j * omega, mu_vt)
        b = mp.pcfd(1.0j * omega, mu_vr)

        result = np.cdouble(
            self.r0 * (mp.fabs(a) ** 2 - mp.exp(2.0 * Delta) * mp.fabs(b) ** 2) / (mp.fabs(a - mp.exp(Delta) * b)) ** 2)
        assert np.imag(result) == 0
        return np.real(result)


class LIFAC:
    def __init__(self,
                 mu_in,  # mean input current
                 d_in,  # diffusion coefficient
                 delta_in,  # jump height of adaptation
                 tau_a_in  # adaptation time constant
                 ):
        self.mu = mu_in
        self.D = d_in
        self.Delta = delta_in
        self.tau_a = tau_a_in
        self.mean_a = self.get_mean_a()
        self.r0 = self.stationary_rate()
        self.lif = LIF(mu_in - self.mean_a, d_in)

    @classmethod
    def from_ini(cls, ini_path):
        config = configparser.ConfigParser()
        config.read(ini_path)

        neuron = config['Neuron']
        assert (neuron['type'] == "LIFAC")

        mu = float(neuron['mu'])
        d = float(neuron['D'])
        delta = float(neuron['Delta'])
        tau_a = float(neuron['tau_a'])

        return cls(mu, d, delta, tau_a)

    def consistency_func(self, mean_a):
        res, err = integrate.quad(integrand, (self.mu - mean_a - 1.0) / np.sqrt(2.0 * self.D),
                                  (self.mu - mean_a) / np.sqrt(2.0 * self.D), epsrel=1e-5)
        r0 = 1.0 / (np.sqrt(pi) * res)
        return self.Delta * self.tau_a * r0

    def get_mean_a(self):
        res = optimize.fixed_point(self.consistency_func,
                                   self.Delta * self.tau_a / (1.0 + self.Delta * self.tau_a) * (self.mu - 0.5))
        return res.item()

    def stationary_rate(self):
        res, err = integrate.quad(integrand, (self.mu - self.mean_a - 1.0) / np.sqrt(2.0 * self.D),
                                  (self.mu - self.mean_a) / np.sqrt(2.0 * self.D), epsrel=1e-5)
        r0 = 1.0 / (np.sqrt(pi) * res)
        return r0

    def integrand_for_stat(self, x):
        return np.real(self.lif.susceptibility_2(-x, x)) * self.Delta ** 2 * self.tau_a ** 2 / (
            1.0 - self.tau_a ** 2 * x ** 2) * self.lif.power_spectrum(x)

    def susceptibility_1(self, omega):
        suscept1 = self.lif.susceptibility_1(omega)
        Omega = self.Delta * self.tau_a / (1.0 - 1.0j * self.tau_a * omega)
        result = np.cdouble(suscept1 / (1.0 + Omega * suscept1))
        return result

    # zeroth order
    # def susceptibility_2(self, omega_1, omega_2):
    #    result = self.lif.susceptibility_2(omega_1, omega_2)
    #    suscept1 = self.lif.susceptibility_1(omega_1 + omega_2)
    #    result = np.cdouble(
    #        result / (1.0 + self.Delta * self.tau_a / (1.0 - 1.0j * self.tau_a * (omega_1 + omega_2))) * (
    #                suscept1 + 2.0 * self.Delta * self.tau_a * self.stationary_rate() * self.lif.susceptibility_2(1e-7,
    #                                                                                                              omega_1 + omega_2)))
    #    return result

    def c_function(self, omega):
        suscept1 = self.lif.susceptibility_1(omega)
        suscept2 = self.lif.susceptibility_2(1e-7, omega)
        Omega = self.Delta * self.tau_a / (1.0 - 1.0j * self.tau_a * omega)
        B = suscept1 + 2.0 * self.Delta * self.tau_a * self.stationary_rate() * suscept2

        return - Omega * B

    def susceptibility_2(self, omega_1, omega_2):
        suscept2 = self.lif.susceptibility_2(omega_1, omega_2)
        suscept1 = self.lif.susceptibility_1(omega_1 + omega_2)
        Omega_sum = self.Delta * self.tau_a / \
            (1.0 - 1.0j * self.tau_a * (omega_1 + omega_2))
        B_sum = suscept1 + 2.0 * self.Delta * self.tau_a * self.stationary_rate() * self.lif.susceptibility_2(1e-7,
                                                                                                              omega_1 + omega_2)
        C_sum = - Omega_sum * B_sum
        Omega_1 = self.Delta * self.tau_a / \
            (1.0 - 1.0j * self.tau_a * (omega_1 + omega_2))
        B_1 = self.lif.susceptibility_1(
            omega_1) + 2.0 * self.Delta * self.tau_a * self.stationary_rate() * self.lif.susceptibility_2(1e-7, omega_1)
        C_1 = -Omega_1 * B_1

        result = np.cdouble(suscept2 / (1.0 - C_sum) + 2.0 *
                            suscept2 * C_1 / ((1 - C_sum) * (1 - C_1)))
        return result
