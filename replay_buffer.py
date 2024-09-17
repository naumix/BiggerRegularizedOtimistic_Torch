import numpy as np

class ReplayBuffer:
    def __init__(self, state_size: int, action_size: int, capacity: int):
        self.states = np.empty((capacity, state_size), dtype=np.float32)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)
        self.next_states = np.empty((capacity, state_size), dtype=np.float32)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        
    def insert(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, mask: float):
        self.states[self.insert_index] = state
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.next_states[self.insert_index] = next_state
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indx = np.random.randint(self.size, size=batch_size)
        states = self.states[indx]
        actions = self.actions[indx]
        rewards = self.rewards[indx]
        masks = self.masks[indx]
        next_states = self.next_states[indx]
        return states, actions, rewards, next_states, masks

    def sample_multibatch(self, batch_size: int, num_batches: int):
        indx = np.random.randint(self.size, size=(num_batches, batch_size))
        states = self.states[indx]
        actions = self.actions[indx]
        rewards = self.rewards[indx]
        masks = self.masks[indx]
        next_states = self.next_states[indx]
        return states, actions, rewards, next_states, masks