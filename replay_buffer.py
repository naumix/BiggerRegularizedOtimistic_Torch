import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size: int, observation_size: int, action_size: int, device: str = 'cpu'):
        self.observations = np.empty((buffer_size, observation_size), dtype=np.float32)
        self.actions = np.empty((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.empty((buffer_size,), dtype=np.float32)
        self.dones = np.empty((buffer_size,), dtype=np.float32)
        self.next_observations = np.empty((buffer_size, observation_size), dtype=np.float32)
        self.size = 0
        self.insert_index = 0
        self.buffer_size = buffer_size
        self.device = device

    def add(self, observation: np.ndarray, next_observation: np.ndarray, action: np.ndarray, reward: float, done: float, infos: dict):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.dones[self.insert_index] = done
        self.next_observations[self.insert_index] = next_observation
        self.insert_index = (self.insert_index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int):
        indx = np.random.randint(self.size, size=batch_size)
        observations = self.observations[indx]
        actions = self.actions[indx]
        rewards = self.rewards[indx]
        dones = self.dones[indx]
        next_observations = self.next_observations[indx]
        return observations, next_observations, actions, rewards, dones

    def sample_multibatch(self, batch_size: int, num_batches: int):
        indx = np.random.randint(self.size, size=(num_batches, batch_size))
        observations = self.to_tensor(self.observations[indx])
        actions = self.to_tensor(self.actions[indx])
        rewards = self.to_tensor(self.rewards[indx])
        dones = self.to_tensor(self.dones[indx])
        next_observations = self.to_tensor(self.next_observations[indx])
        return observations, next_observations, actions, rewards, dones
    
    def to_tensor(self, array: np.ndarray):
        return torch.from_numpy(array).to(self.device)