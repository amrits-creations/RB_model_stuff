import numpy as np


class PredictiveCodingLayer:
    def __init__(self, input_dim, hidden_dim, k1=0.8, k2=0.3, sigma2=1.0, sigma2_td=1.0, alpha=0.1, lam=0.02):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k1 = k1
        self.k2 = k2
        self.sigma2 = sigma2
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
        term3 = (self.k1 / 2) * g_prime

        dr_dt = term1 + term2 - term3
        self.r += dr_dt * dt
        return self.r, bottom_up_error

    def update_weights(self, I, dt=0.1):
        prediction = self.get_prediction()
        bottom_up_error = I - prediction

        hebbian_term = (self.k2 / self.sigma2) * np.dot(bottom_up_error, self.r.T)
        decay_term = self.k2 * self.lam * self.U

        dU_dt = hebbian_term - decay_term
        self.U += dU_dt * dt
        return self.U
