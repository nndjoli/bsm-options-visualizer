import numpy as np
from scipy.stats import norm


class BlackScholes:

    def __init__(self, s, k, r, t, sigma):

        # Parameters:
        self.s = s  # Underlying Price
        self.k = k  # Contract Strike Price
        self.r = r  # Risk-Free Rate
        self.t = t  # Time to Maturity (Years)
        self.sigma = sigma  # Volatility

        # Roots:
        self.d1 = (np.log(self.s / self.k) + (self.r + 0.5 * self.sigma**2) * self.t) / (self.sigma * np.sqrt(self.t))
        self.d2 = self.d1 - (self.sigma * np.sqrt(self.t))

        # Call & Put Prices:
        self.c = self.s * norm.cdf(self.d1) - self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2)
        self.p = -self.s * norm.cdf(-self.d1) + self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2)

        # Delta:
        self.delta_c = norm.cdf(self.d1)
        self.delta_p = norm.cdf(self.d1) - 1

        # Gamma:
        self.gamma = norm.pdf(self.d1) / (self.s * self.sigma * np.sqrt(self.t))

        # Vega:
        self.vega = self.s * norm.pdf(self.d1) * np.sqrt(self.t) / 100  # Divided by 100 for 1% move

        # Theta:
        self.theta_c = (-self.s * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.t)) - self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2)) / 365

        self.theta_p = (-self.s * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.t)) + self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2)) / 365

        # Rho:
        self.rho_c = self.k * self.t * np.exp(-self.r * self.t) * norm.cdf(self.d2) / 100  # Divided by 100 for 1% move
        self.rho_p = -self.k * self.t * np.exp(-self.r * self.t) * norm.cdf(-self.d2) / 100
