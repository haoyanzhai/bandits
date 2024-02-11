import numpy as np


class Environment(object):
    def __init__(self, bandit, agent, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agent = agent
        self.label = label

    def reset(self):
        self.bandit.reset()
        self.agent.reset()

    def run(self, trials=100):
        scores = np.zeros(trials)
        actions = []
        optimal = np.zeros_like(scores)

        for t in range(trials):
            action = self.agent.choose()
            reward, action, is_optimal = self.bandit.pull(action)
            self.agent.observe(reward)

            scores[t] += reward
            actions.append(action)
            if is_optimal:
                optimal[t] += 1

        return scores, actions, optimal
