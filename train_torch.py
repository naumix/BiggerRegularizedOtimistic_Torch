import os

#os.environ['MUJOCO_GL'] = 'egl'

from make_dmc import make_env_dmc
#from stable_baselines3.common.buffers import ReplayBuffer
from replay_buffer import ReplayBuffer
from bro_torch import BRO

import torch
import numpy as np
import random

import wandb

from absl import app, flags
#flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 5, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 25000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 128, 'Mini batch size.')
flags.DEFINE_integer('max_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', 1000000, '.')
flags.DEFINE_integer('start_training', 2500, 'Number of training steps to start training.')
flags.DEFINE_integer('replay_ratio', 2, 'Number of updates per step.')
flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
FLAGS = flags.FLAGS


class flags_:
    #seed = 0
    max_steps = 1000
    batch_size = 128
    start_training = 1001
    replay_ratio = 2
    torch_deterministic = True
    eval_episodes = 5
    eval_interval = 5000
    env_name = 'cheetah-run'
FLAGS = flags_()

def get_seed():
    return np.random.randint(0,1e8)

def get_done(termination, truncation):
        if not termination or truncation:
            done = 0.0
        else:
            done = 1.0
        return done

def sample_multibatch(buffer, batch_size, replay_ratio):
    batches = []
    for i in range(replay_ratio):
        batch = buffer.sample(batch_size)
        batches += [batch]
    return batches

def evaluate(eval_env, agent, eval_episodes: int, temperature: float = 0.0):
    returns = np.zeros(eval_episodes)
    for episode in range(eval_episodes):
        episode_done = False
        observation, _ = eval_env.reset(seed=get_seed())
        while episode_done is False:
            with torch.no_grad():
                action = agent.get_action(torch.from_numpy(observation).unsqueeze(0).to(agent.device), get_log_prob=False, temperature=0.0)
            action = action.detach().cpu().numpy()[0]
            next_observation, reward, termination, truncation, _ = eval_env.step(action)
            returns[episode] += reward
            observation = next_observation
            if termination or truncation:
                episode_done = True
    return {'returns': returns.mean()}
             
def log_to_wandb(step, infos):
    dict_to_log = {'timestep': step}
    for info_key in infos:
        dict_to_log[f'{info_key}'] = infos[info_key]
    wandb.log(dict_to_log, step=step)
      
def main(_):
    SEED = get_seed()
    wandb.init(
        config=FLAGS,
        entity='naumix',
        project='BRO_Torch',
        group=f'{FLAGS.env_name}',
        name=f'BRO_seed:{SEED}_RR:{FLAGS.replay_ratio}'
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    #torch.backends.cudnn.deterministic = FLAGS.torch_deterministic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    env = make_env_dmc(FLAGS.env_name)
    eval_env = make_env_dmc(FLAGS.env_name)
    
    buffer = ReplayBuffer(buffer_size=FLAGS.max_steps, observation_size=env.observation_space.shape[-1], action_size=env.action_space.shape[-1], device=device)
    agent = BRO(env.observation_space.shape[-1], env.action_space.shape[-1], device=device)
    
    observation, _ = env.reset(seed=get_seed())
    for i in range(1, FLAGS.max_steps + 1):
        if i <= FLAGS.start_training:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agent.get_action(torch.from_numpy(observation).unsqueeze(0).to(device), get_log_prob=False, temperature=1.0)
            action = action.detach().cpu().numpy()[0]
        next_observation, reward, termination, truncation, _ = env.step(action)
        done = get_done(termination, truncation)
        buffer.add(observation, next_observation, action, reward, done, {})
        observation = next_observation
        if termination or truncation:
            observation, _ = env.reset(seed=get_seed())
        if i > FLAGS.start_training:
            observations, next_observations, actions, rewards, dones = buffer.sample_multibatch(FLAGS.batch_size, FLAGS.replay_ratio)
            info = agent.update(i, observations, next_observations, actions, rewards, dones)
        if (i % FLAGS.eval_interval) == 0:
            eval_info = evaluate(eval_env, agent, FLAGS.eval_episodes)
            infos = {**info, **eval_info}
            log_to_wandb(i, infos)
            
if __name__ == '__main__':
    app.run(main)

        