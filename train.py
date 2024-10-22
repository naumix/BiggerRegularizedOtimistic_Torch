from make_dmc import _make_env_dmc
from stable_baselines3.common.buffers import ReplayBuffer
from bro_torch import BRO

import torch
import numpy as np

class flags:
    max_steps = 1000
    batch_size = 128
    init_steps = 1001
    replay_ratio = 2

def get_seed():
    return np.random.randint(0,1e8)

def get_done(termination, truncation):
        if not termination or truncation:
            done = 0.0
        else:
            done = 1.0
        return done

def sample_multibatch(buffer, batch_size, replay_ratio):
    states, actions, next_states, dones, rewards = [], [], [], [], []
    for i in range(replay_ratio):
        state, action, next_state, done, reward = buffer.sample(batch_size)
        states += [state]
        actions += [action]
        next_states += [next_state]
        dones += [done]
        rewards += [reward]
    return states, actions, next_states, dones, rewards

FLAGS = flags()
env = _make_env_dmc('cheetah-run', 1)
buffer = ReplayBuffer(FLAGS.max_steps, env.observation_space, env.action_space, 'cpu', handle_timeout_termination=False)
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
    done = get_done(termination, truncation)
    buffer.add(state, next_state, action, reward, done, {})
    state = next_state
    if termination or truncation:
        state, _ = env.reset(seed=get_seed())
    if i > FLAGS.init_steps:
        states, actions, next_states, dones, rewards = sample_multibatch(buffer, FLAGS.batch_size, FLAGS.replay_ratio)
        info = agent.update(i, states, actions, rewards, next_states, dones, FLAGS.replay_ratio) 

    