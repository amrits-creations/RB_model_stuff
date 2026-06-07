from abc import ABC, abstractmethod


class PredictiveCodingNetwork(ABC):
    @abstractmethod
    def get_prediction(self):
        pass

    @abstractmethod
    def update_state(self, I, r_td, dt=0.1):
        pass

    @abstractmethod
    def update_weights(self, I, dt=0.1):
        pass
