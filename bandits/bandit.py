from abc import ABC, abstractmethod

from bayes_opt import BayesianOptimization
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import ParameterGrid, ParameterSampler


class MultiArmedBandit(ABC):
    """
    A Multi-armed Bandit
    """
    def __init__(self, func, params_list_dict):
        self.func = func
        self.params_list_dict = params_list_dict

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pull(self, action):
        pass


class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution
    with provided mean and standard deviation.
    """
    def __init__(
        self,
        func,
        params_list_dict,
        pbounds,
        init_points,
        n_iter,
        random_state,
    ):
        super(GaussianBandit, self).__init__(func, params_list_dict)
        self.pbounds = pbounds
        self.init_points = init_points
        self.n_iter = n_iter
        self.random_state = random_state
        self.optimizer_dict = {}
        self.reset()

    def _get_action_values(self, discrete_params):
        def black_box_function(**continuous_params):
            return self.func(**continuous_params, **discrete_params)
        if str(discrete_params) not in self.optimizer_dict:
            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=self.pbounds,
                # random_state=self.random_state,
                allow_duplicate_points=False,
            )
            init_points = self.init_points
            n_iter = max(self.n_iter - init_points, 0)
            self.optimizer_dict[str(discrete_params)] = optimizer
        else:
            init_points = 0
            n_iter = self.n_iter
        self.optimizer_dict[str(discrete_params)].maximize(
            init_points=init_points, n_iter=n_iter
        )
        return_res_num = max(n_iter, init_points)
        return (
            self.optimizer_dict[str(discrete_params)].max["target"],
            self.optimizer_dict[str(discrete_params)].max["params"],
            self.optimizer_dict[str(discrete_params)].res[-return_res_num:],
        )

    def _update_action_value(self, action):
        action_tuple = list(self.action_values.keys())[action]
        params = {
            key: val
            for key, val in zip(self.params_list_dict.keys(), action_tuple)
        }
        self.action_values[action_tuple], inner_params, conti_action = (
            self._get_action_values(params)
        )
        return (
            self.action_values[action_tuple], inner_params, conti_action
        )

    def reset(self):
        params_grid = ParameterGrid(self.params_list_dict)
        self.action_values = {}
        for params in params_grid:
            params = {key: params[key] for key in self.params_list_dict.keys()}
            self.action_values[tuple(params.values())] = 0

    def pull(self, action):
        action_value, opt_continuous, conti_action = self._update_action_value(
            action
        )
        return (
            action_value,
            (list(self.action_values.keys())[action], opt_continuous),
            conti_action
        )


class GaussianBanditRandomSearch(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution
    with provided mean and standard deviation.
    """
    def __init__(
        self,
        func,
        params_list_dict,
        pbounds,
        max_single_discrete_param_n_iter,
        random_state,
    ):
        super(
            GaussianBanditRandomSearch, self
        ).__init__(func, params_list_dict)
        self.pbounds = pbounds
        self.random_state = random_state
        self.optimizer_dict = {}
        self.max_single_discrete_param_n_iter = (
            max_single_discrete_param_n_iter
        )
        self.reset()

    def _get_action_values(self, discrete_params):
        def black_box_function(**continuous_params):
            return self.func(**continuous_params, **discrete_params)
        if str(discrete_params) not in self.optimizer_dict:
            params_list_dict = {
                key: stats.uniform(self.pbounds[key][0], self.pbounds[key][1])
                for key in self.pbounds.keys()
            }
            param_list = list(
                ParameterSampler(
                    params_list_dict,
                    n_iter=self.max_single_discrete_param_n_iter,
                    # random_state=self.random_state
                )
            )
            self.optimizer_dict[str(discrete_params)] = (
                param_list, 0, -np.inf, -1
            )
        params, idx, opt_val, opt_idx = self.optimizer_dict[
            str(discrete_params)
        ]
        val = black_box_function(**params[idx])
        if val > opt_val:
            opt_val = val
            opt_idx = idx
        idx += 1
        return (opt_val, (params[opt_idx], discrete_params), None)

    def _update_action_value(self, action):
        action_tuple = list(self.action_values.keys())[action]
        params = {
            key: val
            for key, val in zip(self.params_list_dict.keys(), action_tuple)
        }
        self.action_values[action_tuple], inner_params, conti_action = (
            self._get_action_values(params)
        )
        return (
            self.action_values[action_tuple], inner_params, conti_action
        )

    def reset(self):
        params_grid = ParameterGrid(self.params_list_dict)
        self.action_values = {}
        for params in params_grid:
            params = {key: params[key] for key in self.params_list_dict.keys()}
            self.action_values[tuple(params.values())] = 0

    def pull(self, action):
        action_value, opt_continuous, conti_action = self._update_action_value(
            action
        )
        return (
            action_value,
            (list(self.action_values.keys())[action], opt_continuous),
            conti_action
        )


class GaussianBanditOri(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution
    with provided mean and standard deviation.
    """
    def __init__(
        self,
        func,
        params_list_dict,
    ):
        super(GaussianBanditOri, self).__init__(
            func, params_list_dict
        )
        self.reset()

    def reset(self):
        params_grid = ParameterGrid(self.params_list_dict)
        self.action_values = {}
        for params in params_grid:
            params = {key: params[key] for key in self.params_list_dict.keys()}
            self.action_values[tuple(params.values())] = 0

    def pull(self, action):
        action_tuple = list(self.action_values.keys())[action]
        params = {
            key: val
            for key, val in zip(self.params_list_dict.keys(), action_tuple)
        }
        func_value = self.func(**params)
        return (func_value, (action_tuple, None), None)
