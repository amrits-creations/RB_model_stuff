import numpy as np

from .base import PredictiveCodingNetwork


class SingleLayerPCN(PredictiveCodingNetwork):
    def __init__(self, input_dim, hidden_dim, k1=0.8, k2=0.3, sigma2=1.0, sigma2_td=1.0, alpha=0.1, lam=0.02):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k1 = k1
        self.k2 = k2
        self.sigma2 = sigma2 # We name it sigma2 because it's the variance of the noise in the generative model, which is usually denoted as sigma^2.
        self.sigma2_td = sigma2_td
        self.alpha = alpha
        self.lam = lam

        self.r = np.zeros((hidden_dim, 1))
        self.U = np.random.randn(input_dim, hidden_dim) * 0.1

    def get_prediction(self):
        return np.dot(self.U, self.r)

    def update_state(self, I, r_td, dt=0.1):
        if r_td is None: r_td = np.zeros_like(self.r)

        prediction = self.get_prediction()
        bottom_up_error = I - prediction
        top_down_error = r_td - self.r
        g_prime = 2 * self.alpha * self.r

        term1 = (self.k1 / self.sigma2) * np.dot(self.U.T, bottom_up_error)
        term2 = (self.k1 / self.sigma2_td) * top_down_error
        r_regularization = (self.k1 / 2) * g_prime # Renamed this to r_regularization to be more descriptive

        dr_dt = term1 + term2 - r_regularization
        self.r += dr_dt * dt

    def update_weights(self, I, dt=0.1):
        prediction = self.get_prediction()
        bottom_up_error = I - prediction

        hebbian_term = (self.k2 / self.sigma2) * np.dot(bottom_up_error, self.r.T)
        U_regularization = self.k2 * self.lam * self.U # Renamed this to U_regularization to be more descriptive (serves a similar function as r_regularization in the state update - they're both L2 regularization terms)

        dU_dt = hebbian_term - U_regularization
        self.U += dU_dt * dt
