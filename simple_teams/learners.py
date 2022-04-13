import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from .utils import random_argmax


from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from iteration_utilities import unique_everseen

# print(timeit.timeit('from scipy.stats import beta;beta(8,8);', number=10000))


class bayesian_beta_learner:
    def __init__(self, prior={"type": "flat"}):
        """[Initializes a bayesian learner for two-person games, using beta-distributions]]

        Args:
            prior (dict, optional): [Type of prior.]. Defaults to {"type": "flat"}.
        """
        self.prior = prior
        if self.prior["type"] == "flat":
            self.a_b = [1, 1]
            self.beta = beta(self.a_b[0], self.a_b[1])
        if self.prior["type"] == "informative":
            self.a_b = self.prior["a_b"]
            self.beta = beta(self.a_b[0], self.a_b[1])

        self.my_distributions = [self.beta]

    def return_prediction(self, estimate="expected"):
        if estimate == "probabilistic":
            return self.my_distributions[-1].rvs(size=1)
        elif estimate == "expected":
            return self.my_distributions[-1].expect()
        elif estimate == "expected_approx":
            n = 1000
            return (
                random_argmax(self.my_distributions[-1].pdf(np.linspace(0, 1, n))) / n
            )

    def update(self, option_to_add):
        self.a_b[option_to_add] += 1
        self.my_distributions.append(beta(self.a_b[0], self.a_b[1]))

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.my_distributions[-1].pdf(x))
        plt.show()


# class bayesian_gaussian_process:
#     def __init__(self):

#         self.optimizer = BayesianOptimization(
#             f=None,
#             pbounds={"omega": (0.0, 0.99999)},
#             verbose=2,
#             random_state=1,
#         )
#         self.utility = UtilityFunction(
#             kind="ucb", kappa=0.4, xi=0.01
#         )  # kappa = how exploratory

#     def return_new_prediction(self):
#         self.next_point_to_probe = self.optimizer.suggest(self.utility)
#         return self.next_point_to_probe

#     def update_learner(self, target):
#         self.optimizer.register(params=self.next_point_to_probe, target=target)


class bayesian_gaussian_process:
    def __init__(self, window=10, kappa=3, xi=0.0, alpha=1):

        self.attempts = []
        self.distributions = []
        self.predictions = []

        self.window = window
        self.kappa = kappa
        self.xi = xi
        self.alpha = alpha

    def return_prediction(self):
        try:
            for attr in ("current_optimizer", "current_utility"):
                self.__dict__.pop(attr, None)
        except:
            pass

        self.current_optimizer = BayesianOptimization(
            f=None,
            pbounds={"omega": (0.000001, 0.99999)},
            verbose=2,
            random_state=np.int(2 ** 31 * np.random.rand())
            #             random_state=1, # We can't fix a random state here, else we get duplicate point issues :-/
        )
        self.current_utility = UtilityFunction(
            kind="ucb", kappa=self.kappa, xi=self.xi
        )  	# kappa = how exploratory
        	# check function of xi
        	# alpha how tolerant towards noice
        self.current_optimizer._gp.alpha = self.alpha

        if len(self.attempts) <= self.window:
            list_to_register = list(unique_everseen(self.attempts))
            # this way of removing duplicate points might hide problems in the future?
        else:
            list_to_register = list(unique_everseen(self.attempts[-self.window :]))

        for previous_attempt in list_to_register:
            self.current_optimizer.register(
                params=previous_attempt["omega"], target=previous_attempt["target"]
            )

        self.next_point_to_probe = self.current_optimizer.suggest(self.current_utility)

        # keep updates for plotting
        x = np.linspace(0, 1, 100)
        mean, sigma = self.current_optimizer._gp.predict(
            x.reshape(-1, 1), return_std=True
        )
        self.distributions.append((mean, sigma))

        # add maximum
        # print(self.current_optimizer.max)
        try:  # exception for no max at first step. Fix later
            max_omega = self.current_optimizer.max["params"]["omega"]
            self.predictions.append(
                (
                    max_omega,
                    self.current_optimizer._gp.predict(
                        np.array(max_omega).reshape(-1, 1), return_std=False
                    ),
                )
            )
        except:
            pass

        del self.current_optimizer

        return self.next_point_to_probe["omega"]

    def update(self, target):
        if self.next_point_to_probe["omega"] in [x["omega"] for x in self.attempts]:
            self.attempts.append(
                {
                    "omega": self.next_point_to_probe["omega"]
                    + np.random.rand() * 0.0001,
                    "target": target,
                }
            )
        else:
            self.attempts.append(
                {"omega": self.next_point_to_probe["omega"], "target": target}
            )
