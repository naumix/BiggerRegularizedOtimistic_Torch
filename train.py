from make_dmc import _make_env_dmc
from replay_buffer import ReplayBuffer
from bro_torch import BRO

import torch
import numpy as np

def get_seed():
    return np.random.randint(0,1e8)

max_steps = 5

env = _make_env_dmc('cheetah-run', 1)
buffer = ReplayBuffer(env.observation_space.shape[-1], env.action_space.shape[-1], max_steps)
agent = BRO(env.observation_space.shape[-1], env.action_space.shape[-1])

state, _ = env.reset(seed=get_seed())
for i in range(max_steps):
    with torch.no_grad():
        action = agent.get_action(torch.from_numpy(state).unsqueeze(0), get_log_prob=False)
    action = action.detach().numpy()[0]
    next_state, reward, termination, truncation, _ = env.step(action)
    mask = env.get_mask(termination, truncation)
    buffer.insert(state, action, reward, next_state, mask)
    state = next_state
    if termination or truncation:
        state, _ = env.reset()
    