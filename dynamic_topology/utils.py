# import numpy as np
# import torch
#
#
# class ExperienceReplayBuffer(object):
#     def __init__(self, state_dim, action_dim, max_size=int(1e6)):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0
#
#         self.state = np.zeros((max_size, state_dim))
#         self.action = np.zeros((max_size, action_dim))
#         self.next_state = np.zeros((max_size, state_dim))
#         self.reward = np.zeros((max_size, 1))
#         self.not_done = np.zeros((max_size, 1))
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def add(self, state, action, next_state, reward, done):
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.next_state[self.ptr] = next_state
#         self.reward[self.ptr] = reward
#         self.not_done[self.ptr] = 1. - done
#
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def sample(self, batch_size):
#         index = np.random.randint(0, self.size, size=batch_size)
#
#         return (
#             torch.FloatTensor(self.state[index]).to(self.device),
#             torch.FloatTensor(self.action[index]).to(self.device),
#             torch.FloatTensor(self.next_state[index]).to(self.device),
#             torch.FloatTensor(self.reward[index]).to(self.device),
#             torch.FloatTensor(self.not_done[index]).to(self.device)
#         )

import numpy as np
import torch


class ExperienceReplayBuffer:
    def __init__(self, sdim, adim, max_size=int(1e6)):
        self.ptr = 0
        self.size = 0
        self.max = max_size

        self.state      = np.zeros((max_size, sdim))
        self.action     = np.zeros((max_size, adim))
        self.next_state = np.zeros((max_size, sdim))
        self.reward     = np.zeros((max_size, 1))
        self.not_done   = np.zeros((max_size, 1))

    def add(self, s, a, s2, r, done):
        self.state[self.ptr]      = s
        self.action[self.ptr]     = a
        self.next_state[self.ptr] = s2
        self.reward[self.ptr]     = r
        self.not_done[self.ptr]   = 1. - done
        self.ptr  = (self.ptr + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def sample(self, batch, device):
        idx = np.random.randint(0, self.size, size=batch)
        return (torch.FloatTensor(self.state[idx]).to(device),
                torch.FloatTensor(self.action[idx]).to(device),
                torch.FloatTensor(self.next_state[idx]).to(device),
                torch.FloatTensor(self.reward[idx]).to(device),
                torch.FloatTensor(self.not_done[idx]).to(device))

