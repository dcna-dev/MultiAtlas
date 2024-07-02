import math
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
from fake_state import FakeState


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    

class DQNAgent:

    def __init__(self, n_observations, n_actions, batch_size=128, gamma=0.99, 
                 epsilon_start=0.9, epsilon_stop=0.05, epsilon_decay=1000, 
                 tau=0.005, learning_rate=1e-4):
        
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_stop = epsilon_stop
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.learning_rate = learning_rate
        self.steps_done = 0

        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(10000)


    def select_action(self, state):
        sample = random.random()
        epsilon_threshold = self.epsilon_stop + \
                            (self.epsilon_start - self.epsilon_stop) * \
                            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], 
                                dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        #import pdb; pdb.set_trace()
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


def get_reward(state):
    cpu_threshold = 0.8  # CPU threshold for SLA violation
    memory_threshold = 0.8  # Memory threshold for SLA violation
    cpu_slo = 0.7  # CPU SLO (Service Level Objective)
    memory_slo = 0.7  # Memory SLO (Service Level Objective)
    penalty = 0  # Initialize penalty

    if len(state) > 0:
        cpu_violation = max(0, state[0] - cpu_threshold)  # Calculate CPU SLA violation
        cpu_slo_violation = max(0, state[0] - cpu_slo) # Calculate CPU SLO violation
        penalty_cpu_slo = -0.5 if cpu_slo_violation > 0 else 0  # Penalty for CPU SLO violation
        penalty_cpu_sla = -1 if cpu_violation > 0 else 0  # Penalty for SLA violation
        penalty += penalty_cpu_slo + penalty_cpu_sla
    
    if len(state) > 1:
        memory_violation = max(0, state[1] - memory_threshold)  # Calculate Memory SLA violation
        memory_slo_violation = max(0, state[1] - memory_slo) # Calculate Memory SLO violation
        penalty_memory_slo = -0.5 if memory_slo_violation > 0 else 0  # Penalty for Memory SLO violation
        penalty_memory_sla = -1 if memory_violation > 0 else 0  # Penalty for SLA violation
        penalty += penalty_memory_slo + penalty_memory_sla

    # cpu_violation = max(0, state[0] - cpu_threshold)  # Calculate CPU SLA violation
    # memory_violation = max(0, state[1] - memory_threshold)  # Calculate Memory SLA violation

    # cpu_slo_violation = max(0, state[0] - cpu_slo) # Calculate CPU SLO violation
    # memory_slo_violation = max(0, state[1] - memory_slo) # Calculate Memory SLO violation 
      
    # penalty_cpu_slo = -0.5 if cpu_slo_violation > 0 else 0  # Penalty for CPU SLO violation
    # penalty_memory_slo = -0.5 if memory_slo_violation > 0 else 0  # Penalty for Memory SLO violation
    # penalty_cpu_sla = -1 if cpu_violation > 0 else 0  # Penalty for SLA violation
    # penalty_memory_sla = -1 if memory_violation > 0 else 0  # Penalty for SLA violation

    #penalty = penalty_cpu_slo + penalty_memory_slo + penalty_cpu_sla + penalty_memory_sla  # Total penalty

    reward = 1 if penalty == 0 else penalty  # Reward for the agent

    return reward


def plot_data(states, episodes, actions, rewards):
    # Create figure and axes
    fig, ax1 = plt.subplots()

    # Plot state and action on the first Y axis
    ax1.plot(episodes, rewards, color='blue', label='Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Rewards')
    ax1.tick_params(axis='y')

    # Create a second Y axis
    #ax2 = ax1.twinx()

    # Plot slo and sla on the second Y axis
    # ax2.plot(episodes, actions, color='red', label='SLO')
    # ax2.set_ylabel('Action')
    # ax2.tick_params(axis='y')

    # Add legend
    #lines, labels = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    num_episodes = 10000

    inital_state = [0.8]  # Dummy initial state
    state = torch.tensor(inital_state, dtype=torch.float32).unsqueeze(0)

    agent = DQNAgent(len(inital_state), 3)

    states = []
    actions = []
    episodes = []
    rewards = []

    for i_episode in range(num_episodes):
        action = agent.select_action(state)
        next_state = FakeState().get_state(state.tolist()[0], action.tolist()[0][0])
        reward = get_reward(next_state)

        if i_episode % 10 == 0:
            states.append(state.tolist()[0])
            actions.append(action.tolist()[0][0])
            episodes.append(i_episode)
            rewards.append(reward)
        print(f"Episode: {i_episode}, State: {state.tolist()[0]},  Action: {action.tolist()[0][0]}, Next State: {next_state}, Reward: {reward}")
        #import pdb; pdb.set_trace()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        agent.memory.push(state, action, next_state, reward)

        state = next_state
        agent.optimize_model()

        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()
        for key in target_net_state_dict:
            target_net_state_dict[key] = agent.tau * policy_net_state_dict[key] + (1 - agent.tau) * target_net_state_dict[key]
        agent.target_net.load_state_dict(target_net_state_dict)

    plot_data(states, episodes, actions, rewards)