from make_dmc import _make_env_dmc
from replay_buffer import ReplayBuffer
from bro_torch import BRO

import torch
import numpy as np

class flags:
    max_steps = 1000
    batch_size = 128
    init_steps = 998
    replay_ratio = 2

def get_seed():
    return np.random.randint(0,1e8)

FLAGS = flags()

env = _make_env_dmc('cheetah-run', 1)
buffer = ReplayBuffer(env.observation_space.shape[-1], env.action_space.shape[-1], FLAGS.max_steps)
agent = BRO(env.observation_space.shape[-1], env.action_space.shape[-1])

state, _ = env.reset(seed=get_seed())
for i in range(FLAGS.max_steps):
    if i <= FLAGS.init_steps:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            action = agent.get_action(torch.from_numpy(state).unsqueeze(0), get_log_prob=False)
        action = action.detach().numpy()[0]
    next_state, reward, termination, truncation, _ = env.step(action)
    mask = env.get_mask(termination, truncation)
    buffer.insert(state, action, reward, next_state, mask)
    state = next_state
    if termination or truncation:
        state, _ = env.reset()
    if i > FLAGS.init_steps:
        states, actions, rewards, next_states, masks = buffer.sample_multibatch(FLAGS.batch_size, FLAGS.replay_ratio)
        info = agent.update(i, states, actions, rewards, next_states, masks, FLAGS.replay_ratio) 

    