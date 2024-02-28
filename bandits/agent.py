import numpy as np


class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step. The
    action is chosen using a strategy based on the history of prior actions
    and outcome observations.
    """
    def __init__(self, bandit, policy, prior=0, gamma=None):
        self.policy = policy
        self.k = len(bandit.action_values)
        self.prior = prior
        self.gamma = gamma
        self._value_estimates = prior*np.ones(self.k)
        self.action_attempts = np.zeros(self.k)
        self.t = 0
        self.last_action = None

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._value_estimates[:] = self.prior
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1

        if self.gamma is None:
            g = 1 / self.action_attempts[self.last_action]
        else:
            g = self.gamma
        q = self._value_estimates[self.last_action]

        self._value_estimates[self.last_action] += g*(reward - q)
        self.t += 1

    @property
    def value_estimates(self):
        return self._value_estimates


class GradientAgent(Agent):
    """
    The Gradient Agent learns the relative difference between actions instead
    of determining estimates of reward values. It effectively learns a
    preference for one action over another.
    """
    def __init__(
        self,
        bandit,
        policy,
        prior=0,
        alpha=0.1,
        avg_coef=None,
        baseline=True,
        increase_rate=0.01,
    ):
        super(GradientAgent, self).__init__(bandit, policy, prior)
        self.alpha = alpha
        self.adaptive_alpha = self.alpha
        self.increase_rate = increase_rate
        self.increase_value = self.alpha * self.increase_rate
        self.baseline = baseline
        self.average_reward = 0
        self.avg_coef = avg_coef

    def __str__(self):
        return 'g/\u03B1={}, bl={}'.format(self.alpha, self.baseline)

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1

        if self.baseline:
            # self.average_reward = max(reward, self.average_reward)
            diff = reward - self.average_reward
            avg_coef = (
                self.avg_coef if self.avg_coef
                else 1/np.sum(self.action_attempts)
            )
            self.average_reward += avg_coef * diff

        pi = (
            np.exp(self.value_estimates) / np.sum(np.exp(self.value_estimates))
        )
        print(reward)

        ht = self.value_estimates[self.last_action]
        ht += (
            self.adaptive_alpha
            * (reward - self.average_reward)*(1-pi[self.last_action])
        )
        self._value_estimates -= (
            self.adaptive_alpha*(reward - self.average_reward)*pi
        )
        self._value_estimates[self.last_action] = ht
        self.t += 1
        self.adaptive_alpha += self.increase_value
        self.adaptive_alpha = min(self.adaptive_alpha, 1)
        print(self.adaptive_alpha)

    def reset(self):
        super(GradientAgent, self).reset()
        self.average_reward = 0
