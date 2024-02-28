import time
import numpy as np
from collections import defaultdict


class Environment(object):
    def __init__(self, bandit, agent, label, random_state):
        self.bandit = bandit
        self.agent = agent
        self.label = label
        self.random_state = random_state
        np.random.seed(random_state)
        self.reset()

    def reset(self):
        self.bandit.reset()
        self.agent.reset()

    def run(self, trials, file=None, stop_iter=70):
        scores = np.zeros(trials)
        actions = []

        queue = []
        counter = defaultdict(int)
        for t in range(trials):
            print(t)
            start = time.time()
            action = self.agent.choose()
            print(list(self.bandit.action_values.keys())[action])
            reward, action, conti_action = self.bandit.pull(action)
            self.agent.observe(reward)
            end = time.time()
            if file is not None:
                file.write(
                    f"({self.random_state}, {t}, {action}, "
                    f"{conti_action}, {reward}, {end-start})\n"
                )

            scores[t] += reward
            actions.append(action)
            if len(queue) == 1000:
                key = queue.pop(0)
                counter[key] -= 1
                if counter[key] < 0:
                    raise ValueError(f"counter[{key}] < 0")
            new_key = str(action[0])
            print(action, reward)
            queue.append((new_key, f'{reward:.2f}'))
            counter[(new_key, f'{reward:.2f}')] += 1
            if max(counter.values()) >= stop_iter:
                print(
                    f"stop because of {stop_iter}"
                    f"same actions with same reward"
                )
                break

        return scores, actions
