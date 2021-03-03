#!/usr/bin/env python3

from math import pi
import numpy as np
import mpmath as mp
from scipy import optimize
from scipy import integrate
from scipy import special

mp.dps = 15
mp.pretty = True


def integrand(x):
    return np.exp(x * x) * special.erfc(x)


def firing_rate_signal_linear(neuron, t, eps, f):
    omega = 2 * pi * f
    stat = neuron.stationary_rate()
    lr = eps * np.fabs(neuron.susceptibility_1(omega)) * np.cos(omega * t - np.arg(neuron.susceptibility_1(omega)))
    result = stat + lr
    assert np.im(result) == 0
    return np.re(result)


def firing_rate_signal_linear_two_sigs(neuron, t, alpha, f1, beta, f2):
    omega1 = 2 * pi * f1
    omega2 = 2 * pi * f2
    stat = neuron.stationary_rate()
    lr = alpha * np.absolute(neuron.susceptibility_1(omega1)) * np.cos(
        omega1 * t - np.angle(neuron.susceptibility_1(omega1))) + beta * np.absolute(
        neuron.susceptibility_1(omega2)) * np.cos(omega2 * t - np.angle(neuron.susceptibility_1(omega2)))
    result = stat + lr
    # assert np.imag(result) == 0
    return np.real(result)


def firing_rate_signal_nonlinear(neuron, t, eps, f):
    omega = 2 * pi * f
    stat = neuron.stationary_rate() + 0.5 * eps ** 2 * neuron.susceptibility_2(omega, -omega)
    lr = eps * np.absolute(neuron.susceptibility_1(omega)) * np.cos(
        omega * t - np.angle(neuron.susceptibility_1(omega)))
    hh = 0.5 * eps ** 2 * np.absolute(neuron.susceptibility_2(omega, omega)) * np.cos(
        2 * omega * t - np.angle(neuron.susceptibility_2(omega, omega)))
    result = stat + lr + hh
    # assert np.imag(result) == 0
    return np.real(result)


def firing_rate_signal_nonlinear_two_sigs(neuron, t, alpha, f1, beta, f2):
    omega1 = 2 * pi * f1
    omega2 = 2 * pi * f2
    stat = neuron.stationary_rate() \
           + 0.5 * alpha ** 2 * neuron.susceptibility_2(omega1, -omega1) \
           + 0.5 * alpha ** 2 * neuron.susceptibility_2(omega1, -omega1)
    lr = alpha * np.absolute(neuron.susceptibility_1(omega1)) * np.cos(
        omega1 * t - np.angle(neuron.susceptibility_1(omega1))) + beta * np.absolute(
        neuron.susceptibility_1(omega2)) * np.cos(omega2 * t - np.angle(neuron.susceptibility_1(omega2)))
    hh = 0.5 * alpha ** 2 * np.absolute(neuron.susceptibility_2(omega1, omega1)) * np.cos(
        2 * omega1 * t - np.angle(neuron.susceptibility_2(omega1, omega1))) + 0.5 * beta ** 2 * np.absolute(
        neuron.susceptibility_2(omega2, omega2)) * np.cos(
        2 * omega2 * t - np.angle(neuron.susceptibility_2(omega2, omega2)))
    mr = alpha * beta * (np.absolute(neuron.susceptibility_2(omega1, omega2)) * np.cos(
        (omega1 + omega2) * t - np.angle(neuron.susceptibility_2(omega1, omega2))) + np.absolute(
        neuron.susceptibility_2(omega1, -omega2)) * np.cos(
        (omega1 - omega2) * t - np.angle(neuron.susceptibility_2(omega1, -omega2))))
    result = stat + lr + hh + mr
    # assert np.imag(result) == 0
    return np.real(result)


class PIF:
    def __init__(self, mu, d):
        self.mu = mu
        self.D = d

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

    def susceptibility_1(self, omega):
        result = self.pif.susceptibility_1(omega)
        result = np.real(np.cdouble(result)) + 1.0j * np.imag(np.cdouble(result))
        result = result / (1.0 + result * self.Delta * self.tau_a / (1.0 - 1.0j * self.tau_a * omega))
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
        b = mp.exp((2.0 * self.mu - 1.0) / (4.0 * self.D)) * mp.pcfd(1.0j * omega - 1.0, self.mu / mp.sqrt(self.D))
        c = mp.pcfd(1.0j * omega, (self.mu - 1.0) / mp.sqrt(self.D))
        d = mp.exp((2.0 * self.mu - 1.0) / (4.0 * self.D)) * mp.pcfd(1.0j * omega, self.mu / mp.sqrt(self.D))

        result = np.cdouble(alpha * (a - b) / (c - d))
        return result

    def susceptibility_2(self, omega_1, omega_2):
        alpha = self.r0 * (1.0 - 1.0j * omega_1 - 1.0j * omega_2) * (1.0j * omega_1 + 1.0j * omega_2) / (
                2.0 * self.D * (1.0j * omega_1 - 1.0) * (1.0j * omega_2 - 1.0))
        a = mp.pcfd(1.0j * omega_1 + 1.0j * omega_2 - 2.0, (self.mu - 1.0) / mp.sqrt(self.D))
        b = mp.exp((2.0 * self.mu - 1.0) / (4.0 * self.D)) * mp.pcfd(1.0j * omega_1 + 1.0j * omega_2 - 2.0,
                                                                     self.mu / mp.sqrt(self.D))
        c = mp.pcfd(1.0j * omega_1 + 1.0j * omega_2, (self.mu - 1.0) / mp.sqrt(self.D))
        d = mp.exp((2.0 * self.mu - 1.0) / (4.0 * self.D)) * mp.pcfd(1.0j * omega_1 + 1.0j * omega_2,
                                                                     self.mu / mp.sqrt(self.D))

        beta = (1.0j * omega_1 + 1.0j * omega_2) / (2.0 * mp.sqrt(self.D))
        a_2 = (self.susceptibility_1(omega_1) / (1.0j * omega_2 - 1.0) + self.susceptibility_1(omega_2) / (
                1.0j * omega_1 - 1.0)) * mp.pcfd(1.0j * omega_1 + 1.0j * omega_2 - 1.0,
                                                 (self.mu - 1.0) / mp.sqrt(self.D))
        a_3 = (self.susceptibility_1(omega_1) / (1.0j * omega_2 - 1.0) + self.susceptibility_1(omega_2) / (
                1.0j * omega_1 - 1.0)) * mp.exp((2 * self.mu - 1.0) / (4.0 * self.D)) * mp.pcfd(
            1.0j * omega_1 + 1.0j * omega_2 - 1.0, self.mu / mp.sqrt(self.D))

        result = np.cdouble(alpha * (a - b) / (c - d) + beta * a_2 / (c - d) - beta * a_3 / (c - d))
        return result

    def power_spectrum(self, omega):
        mu_vt = (self.mu - 1.0) / (mp.sqrt(self.D))
        mu_vr = self.mu / (mp.sqrt(self.D))
        Delta = (2.0 * self.mu - 1.0) / (4.0 * self.D)
        a = mp.pcfd(1.0j * omega, mu_vt)
        b = mp.pcfd(1.0j * omega, mu_vr)

        result = np.cdouble(
            self.r0 * (mp.fabs(a) ** 2 - mp.exp(2.0 * Delta) * mp.fabs(b) ** 2) / (mp.fabs(a - mp.exp(Delta) * b)) ** 2)
        assert np.im(result) == 0
        return np.re(result)


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
        return np.re(self.lif.susceptibility_2(-x, x)) * self.Delta ** 2 * self.tau_a ** 2 / (
                1.0 - self.tau_a ** 2 * x ** 2) * self.lif.power_spectrum(x)

    def susceptibility_1(self, omega):
        suscept1 = self.lif.susceptibility_1(omega)
        Omega = self.Delta * self.tau_a / (1.0 - 1.0j * self.tau_a * omega)
        result = np.cdouble(suscept1 / (1.0 + Omega * suscept1))

        # suscept1 = self.lif.susceptibility_1(omega)
        # suscept2 = self.lif.susceptibility_2(1e-13, omega)
        # Omega = self.Delta * self.tau_a / (1.0 - 1.0j * self.tau_a * omega)
        # B = suscept1 + 2.0*self.get_mean_a()*suscept2
        # result = np.cdouble(B/(1.0 + Omega * B))

        # suscept_1_lif = self.lif.susceptibility_1(omega)
        # suscept_2_lif = self.lif.susceptibility_2(1e-7, omega)
        # A = 2 * pi * self.r0 + 2 * pi * self.mean_a * self.lif.susceptibility_1(
        #    1e-7) + 2.0 * pi * self.mean_a * self.mean_a * self.lif.susceptibility_2(1e-7, 1e-7)
        # B = suscept_1_lif + 2.0 * self.mean_a * suscept_2_lif
        # C = - self.Delta * self.tau_a / (1 - 1j * self.tau_a * omega) * B
        # result = np.cdouble(
        #    B / (1 - C) - 2 * (1 / (2 * pi) * suscept_2_lif * self.Delta * self.tau_a * A) / ((1 - C) * (
        #            1 - self.Delta * self.tau_a * (
        #            self.lif.susceptibility_1(1e-7) + 2.0 * self.mean_a * self.lif.susceptibility_2(1e-7,
        #                                                                                            1e-7)))))
        return result

    # def susceptibility_2(self, omega_1, omega_2):
    #    result = self.lif.susceptibility_2(omega_1, omega_2)
    #    suscept1 = self.lif.susceptibility_1(omega_1 + omega_2)
    #    result = np.cdouble(
    #        result / (1.0 + self.Delta * self.tau_a / (1.0 - 1.0j * self.tau_a * (omega_1 + omega_2))) * (
    #                suscept1 + 2.0 * self.Delta * self.tau_a * self.stationary_rate() * self.lif.susceptibility_2(1e-7,
    #                                                                                                              omega_1 + omega_2)))
    #    return result

    def susceptibility_2(self, omega_1, omega_2):
        suscept2 = self.lif.susceptibility_2(omega_1, omega_2)
        suscept1 = self.lif.susceptibility_1(omega_1 + omega_2)
        Omega_sum = self.Delta * self.tau_a / (1.0 - 1.0j * self.tau_a * (omega_1 + omega_2))
        B_sum = suscept1 + 2.0 * self.Delta * self.tau_a * self.stationary_rate() * self.lif.susceptibility_2(1e-7,
                                                                                                              omega_1 + omega_2)
        C_sum = - Omega_sum * B_sum
        Omega_1 = self.Delta * self.tau_a / (1.0 - 1.0j * self.tau_a * (omega_1 + omega_2))
        B_1 = self.lif.susceptibility_1(
            omega_1) + 2.0 * self.Delta * self.tau_a * self.stationary_rate() * self.lif.susceptibility_2(1e-7, omega_1)
        C_1 = -Omega_1 * B_1

        result = np.cdouble(suscept2 / (1.0 - C_sum) + 2.0 * suscept2 * C_1 / ((1 - C_sum) * (1 - C_1)))
        return result
