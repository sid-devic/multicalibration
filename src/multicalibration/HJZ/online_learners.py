import numpy as np

class Hedge:
    def __init__(self, n_actions, learning_rate=0.5, f0=None):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.weights = np.ones(n_actions) / n_actions
        if f0 is not None: self.weights = f0

    def predict(self):
        return np.random.choice(self.n_actions, p=self.weights)

    def update(self, loss_vector):
        self.weights *= np.exp(-self.learning_rate * loss_vector)
        ee = self.learning_rate * loss_vector
        self.weights /= np.sum(self.weights)


class OnlineGradientDescent:
    def __init__(self, n_actions, learning_rate=0.5, f0=None):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.weights = np.ones(n_actions) / n_actions
        if f0 is not None: self.weights = f0

    def predict(self):
        return np.random.choice(self.n_actions, p=self.weights)

    def update(self, loss_vector):
        self.weights -= self.learning_rate * loss_vector
        self.weights = self.project(self.weights)

    def project(self, weights):
        """Project weights onto the simplex."""
        if np.sum(weights) <= 1 and np.alltrue(weights >= 0):
            return weights  # already a probability distribution
        u = np.sort(weights)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(self.n_actions) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        return np.maximum(weights - theta, 0)


class MLProd:
    def __init__(self, n_actions, learning_rate, f0=None):
        self.n_actions = n_actions
        if np.isscalar(learning_rate):
            self.learning_rate = np.full(n_actions, learning_rate)
        else:
            assert (
                len(learning_rate) == n_actions
            ), "Need a learning rate for each action"
            self.learning_rate = np.array(learning_rate)
        self.w = np.ones(n_actions) / n_actions
        self.weights = self.w
        
        if f0 is not None:
            self.weights = f0
            self.w = f0

    def predict(self):
        return np.random.choice(self.n_actions, p=self.weights)

    def update(self, loss_vector):
        loss_vector = np.array(loss_vector)
        expected_loss = np.dot(self.weights, loss_vector)
        self.w *= np.clip(1 - self.learning_rate * loss_vector, 1e-8, None)
        self.w /= np.sum(self.w)
        self.weights = self.w


class OptimisticHedge(Hedge):
    def __init__(self, n_actions, learning_rate=0.5, optimism=0.5, f0=None):
        super().__init__(n_actions, learning_rate, f0=f0)
        self.last_loss_vector = 0

    def update(self, loss_vector):
        adjusted_loss = 2 * loss_vector - self.last_loss_vector

        self.weights *= np.exp(-self.learning_rate * adjusted_loss)
        self.weights /= np.sum(self.weights)

        self.last_loss_vector = loss_vector
