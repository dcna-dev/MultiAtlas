from typing import List
import gym
from gym import spaces
import numpy as np


class CloudEnv(gym.Env):
    def __init__(self, provider, memory_slo, cpu_slo):
        super(CloudEnv, self).__init__()

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(3)  # Three possible actions:  scale down, do nothing, scale up
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)  # Memory and CPU utilization

        # Set the initial state
        self.provider = provider
        self.memory_slo = memory_slo
        self.cpu_slo = cpu_slo

    def step(self, action):
        # Update the state based on the action
        if action == 0:  # Scale down
            self.provider.scale_down()
        elif action == 1:  # Do nothing
            pass
        elif action == 2:  # Scale up
            self.provider.scale_up()

        # Calculate the reward based on the current state
        memory_usage = self.provider.get_vm_memory_percent_usage()
        cpu_usage = self.provider.get_vm_cpu_percent_usage()
        if memory_usage > self.memory_slo:
            reward -= 1
        if cpu_usage > self.cpu_slo:
            reward -= 1
        if memory_usage < self.memory_slo/2:
            reward -= 0.5
        if cpu_usage < self.cpu_slo/2:
            reward -= 0.5
        else:
            reward = 1

        done = False
        info = {}

        # Return the next state, reward, and done flag
        next_state = np.array([memory_usage, cpu_usage], dtype=np.float32)
        return next_state, reward, done, info

    def render(self):
        pass

    def reset(self, provider):
        # Reset the state to the initial state
        self.provider = provider