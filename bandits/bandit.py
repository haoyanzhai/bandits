import os
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from abc import ABC, abstractmethod
from sklearn.model_selection import ParameterGrid


class MultiArmedBandit(ABC):
    """
    A Multi-armed Bandit
    """
    def __init__(self, func, params_list_dict, pbounds):
        self.func = func
        self.params_list_dict = params_list_dict
        self.pbounds = pbounds
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pull(self, action):
        pass


class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    """
    def __init__(self, func, params_list_dict, pbounds, init_points, n_iter):
        super(GaussianBandit, self).__init__(func, params_list_dict, pbounds)
        self.optimal = None
        self.init_points = init_points
        self.n_iter = n_iter
        self.reset()

    def _get_action_values(self, discrete_params):
        def black_box_function(**continuous_params):
            return self.func(**continuous_params, **discrete_params)
        log_file = f"./logs/logs_{discrete_params}.log"
        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=self.pbounds,
            random_state=1,
            allow_duplicate_points=True
        )
        if os.path.exists(log_file + '.json'):
            load_logs(optimizer, logs=[log_file + '.json'])
            init_points = 0
        else:
            init_points = self.init_points
            logger = JSONLogger(path=log_file)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(init_points=init_points, n_iter=self.n_iter)
        print(optimizer.max, discrete_params)
        return optimizer.max["target"], optimizer.max["params"]

    def _update_action_value(self, action):
        action_tuple = list(self.action_values.keys())[action]
        params = {
            key: val
            for key, val in zip(self.params_list_dict.keys(), action_tuple)
        }
        print(params)
        self.action_values[action_tuple], inner_params = (
            self._get_action_values(params)
        )
        self.optimal = np.argmax(self.action_values)
        return self.action_values[action_tuple], inner_params

    def reset(self):
        if os.path.exists("./logs"):
            os.system("rm -rf ./logs")
        os.makedirs("./logs")
        params_grid = ParameterGrid(self.params_list_dict)
        self.action_values = {}
        for params in params_grid:
            params = {key: params[key] for key in self.params_list_dict.keys()}
            self.action_values[tuple(params.values())] = (
                self._get_action_values(params)
            )
        self.optimal = np.argmax(self.action_values)

    def pull(self, action):
        discrete, continuous = self._update_action_value(action)
        return (
            discrete,
            (list(self.action_values.keys())[action], continuous),
            action == self.optimal
        )
