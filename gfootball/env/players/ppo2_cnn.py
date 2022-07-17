# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Player from PPO2 cnn checkpoint.

Example usage with play_game script:
python3 -m gfootball.play_game \
    --players "ppo2_cnn:left_players=1,checkpoint=$YOUR_PATH,policy=$POLICY"

$POLICY should be one of: cnn, impala_cnn, gfootball_impala_cnn.
"""

from gfootball.env import football_action_set
from gfootball.env import player_base

import gym
import joblib
import numpy as np
import tensorflow.compat.v1 as tf

#football_syw packages
from .agent import FootballTrainer
import copy

def _t2n(x):
    return x.detach().cpu().numpy()

def get_offside(obs):
        ball = np.array(obs['ball'][:2])
        ally = np.array(obs['left_team'])
        enemy = np.array(obs['right_team'])

        if obs['game_mode'] != 0:
            last_loffside = np.zeros(11, np.float32)
            last_roffside = np.zeros(11, np.float32)
            return np.zeros(11, np.float32), np.zeros(11, np.float32)

        need_recalc = False
        effective_ownball_team = -1
        effective_ownball_player = -1

        if obs['ball_owned_team'] > -1:
            effective_ownball_team = obs['ball_owned_team']
            effective_ownball_player = obs['ball_owned_player']
            need_recalc = True
        else:
            ally_dist = np.linalg.norm(ball - ally, axis=-1)
            enemy_dist = np.linalg.norm(ball - enemy, axis=-1)
            if np.min(ally_dist) < np.min(enemy_dist):
                if np.min(ally_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 0
                    effective_ownball_player = np.argmin(ally_dist)
            elif np.min(enemy_dist) < np.min(ally_dist):
                if np.min(enemy_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 1
                    effective_ownball_player = np.argmin(enemy_dist)


        left_offside = np.zeros(11, np.float32)
        right_offside = np.zeros(11, np.float32)

        if effective_ownball_team == 0:
            right_xs = [obs['right_team'][k][0] for k in range(1, 11)]
            right_xs = np.array(right_xs)
            right_xs.sort()

            for k in range(1, 11):
                if obs['left_team'][k][0] > right_xs[-1] and k != effective_ownball_player \
                   and obs['left_team'][k][0] > 0.0:
                    left_offside[k] = 1.0
        else:
            left_xs = [obs['left_team'][k][0] for k in range(1, 11)]
            left_xs = np.array(left_xs)
            left_xs.sort()

            for k in range(1, 11):
                if obs['right_team'][k][0] < left_xs[0] and k != effective_ownball_player \
                   and obs['right_team'][k][0] < 0.0:
                    right_offside[k] = 1.0

        return left_offside, right_offside

def raw2vec(raw_obs):
        obs = []

        ally = np.array(raw_obs['left_team'])
        ally_d = np.array(raw_obs['left_team_direction'])
        enemy = np.array(raw_obs['right_team'])
        enemy_d = np.array(raw_obs['right_team_direction'])
        ball = np.array(raw_obs['ball'])
        ball_d = np.array(raw_obs['ball_direction'])
        lo, ro = get_offside(raw_obs)
        
        obs.extend(ally.flatten())                          # shape = 22
        obs.extend(ally_d.flatten())                        # shape = 44    
        obs.extend(enemy.flatten())                         # shape = 66
        obs.extend(enemy_d.flatten())                       # shape = 88
        obs.extend(ball.flatten())                          # shape = 91    
        obs.extend(ball_d.flatten())                        # shape = 94
        
        if raw_obs['ball_owned_team'] == -1:
            obs.extend([1, 0, 0])                           # shape = 97
        elif raw_obs['ball_owned_team'] == 0:
            obs.extend([0, 1, 0])                           # shape = 97
        elif raw_obs['ball_owned_team'] == 1:
            obs.extend([0, 0, 1])                           # shape = 97

        obs.extend(np.zeros(11))                            # shape = 108
        game_mode = np.zeros(7)
        game_mode[raw_obs['game_mode']] = 1
        obs.extend(game_mode)                               # shape = 115
        obs.extend(np.zeros(10))                            # shape = 125
        obs.extend(np.zeros(1))                             # shape = 126
        obs.extend(raw_obs['left_team_tired_factor'])    # shape = 137
        obs.extend(raw_obs['left_team_yellow_card'])     # shape = 148
        obs.extend(raw_obs['left_team_active'])          # shape = 159
        obs.extend(lo)                                      # shape = 170
        obs.extend(ro)                                      # shape = 181
        obs.extend(np.zeros(11))                            # shape = 192
        obs.extend(np.zeros(22))                            # shape = 214
        obs.extend(np.zeros(22))                            # shape = 236
        obs.extend(np.zeros(2))                             # shape = 238

        steps_left = raw_obs['steps_left']
        obs.extend([1.0 * steps_left / 3001])               # shape = 239
        if steps_left > 1500:
            steps_left -= 1501                    
        steps_left = 1.0 * min(steps_left, 300.0) 
        steps_left /= 300.0
        obs.extend([steps_left])                            # shape = 240

        score_ratio = 1.0 * (raw_obs['score'][0] - raw_obs['score'][1])
        score_ratio /= 5.0
        score_ratio = min(score_ratio, 1.0)
        score_ratio = max(-1.0, score_ratio)
        obs.extend([score_ratio])                           # shape = 241
        obs.extend(np.zeros(27))                            # shape = 268
        
        me = ally[int(raw_obs['active'])]
        ball = raw_obs['ball'][:2]
        ball_dist = np.linalg.norm(me - ball)
        enemy_dist = np.linalg.norm(me - enemy, axis=-1)
        to_enemy = enemy - me
        to_ally = ally - me
        to_ball = ball - me
        
        obs[97+raw_obs['active']] = 1             
        sticky = raw_obs['sticky_actions'][:10]
        obs[115:125] = sticky                       
        ball_dist = 1 if ball_dist > 1 else ball_dist
        obs[125] = ball_dist 
        obs[181:192] = enemy_dist
        to_ally[:, 0] /= 2
        obs[192:214] = to_ally.flatten()
        to_enemy[:, 0] /= 2
        obs[214:236] = to_enemy.flatten()
        to_ball[0] /= 2
        obs[236:238] = to_ball.flatten()
            
        return np.array(obs)

def collect(obs_list):
    raw_o = obs_list
    obs = raw2vec(raw_o) 
    obs = obs.reshape(1,1,268)
    share_obs = obs.copy()
    share_obs[159:] = 0

    # available actions
    action_size = 33
    avail_size = 20     # buildin ai
    avail_actions = np.ones([1, 1, action_size])
    avail_actions[0, :, avail_size:] = 0
    rewards = []
    infos, dones = [], []
    return obs, share_obs, rewards, dones, infos, avail_actions

# set configs
trainer = FootballTrainer(None)
trainer.driver.trainer.prep_rollout()
rnn_shape = [1,1,1,256]
eval_rnn_states = [np.zeros(rnn_shape, dtype=np.float32) for i in range (11)]
eval_rnn_states_critic = [np.zeros(rnn_shape, dtype=np.float32) for i in range (11)]
trainer.__setattr__('eval_rnn_states',eval_rnn_states)
trainer.__setattr__('eval_rnn_states_critic',eval_rnn_states_critic)

def my_controller(obs_list, i):

    # initialize the configs if a game ends
    if obs_list['steps_left'] == 3000:
        trainer.eval_rnn_states = [np.zeros(rnn_shape, dtype=np.float32) for i in range (11)]
        trainer.eval_rnn_states_critic = [np.zeros(rnn_shape, dtype=np.float32) for i in range (11)]

    obs_list['controlled_player_index'] = i
    # index preprocess
    idx = obs_list['controlled_player_index'] % 11
    del obs_list['controlled_player_index']

    # goalkeeper
    if idx == 0: 
        # return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return 0
    idx = idx - 1
    
    #get actions
    eval_masks = np.ones((1, 1, 1), dtype=np.float32) #(n_eval_rollout_threads,agent_num,1)
    eval_obs, eval_share_obs , _ , __, ___, eval_available_actions = collect(obs_list)
    _, eval_action, _, trainer.eval_rnn_states[idx], _ = \
        trainer.driver.trainer.algo_module.get_actions(np.concatenate(eval_share_obs),
                                                np.concatenate(eval_obs),
                                                np.concatenate(trainer.eval_rnn_states[idx]),
                                                np.concatenate(trainer.eval_rnn_states_critic[idx]),
                                                np.concatenate(eval_masks),
                                                np.concatenate(eval_available_actions),
                                                deterministic=True)
    trainer.eval_rnn_states[idx] = np.array(np.split(_t2n(trainer.eval_rnn_states[idx]), 1))

    return_action = list(np.zeros(20,dtype=int))
    return_action[eval_action.squeeze(1).item()] = 1
    return_idx = 0
    for i in range(len(return_action)):
      if return_action[i] == 1:
        return_idx = i
        break

    return return_idx


class Player(player_base.PlayerBase):
  """An agent loaded from PPO2 cnn model checkpoint."""

  def __init__(self, player_config, env_config):
    player_base.PlayerBase.__init__(self, player_config)

    self._action_set = (env_config['action_set']
                        if 'action_set' in env_config else 'default')
    self._player_prefix = 'player_{}'.format(player_config['index'])
    stacking = 4 if player_config.get('stacked', True) else 1
    policy = player_config.get('policy', 'cnn')
    self._stacker = ObservationStacker(stacking)

  def take_action_bak(self, observation):

    obs_len = len(observation)
    action = my_controller(observation)
    idx = 0
    for i in range(len(action)):
      if action[i] == 1:
        idx = i
        break
    
    
    actions = [football_action_set.action_set_dict[self._action_set][idx]]
    return actions

  def take_action(self, observation):
    
    actions = []
    obs_len = len(observation)
    if obs_len == 1:
      action = my_controller(observation[0], 0)
      actions = [football_action_set.action_set_dict[self._action_set][action]]
    else:
      for idx in range( obs_len ):
        action = my_controller(observation[idx], idx)
        actions.append(football_action_set.action_set_dict[self._action_set][action])

    return actions

  def reset(self):
    self._stacker.reset()


def _load_variables(load_path, sess, prefix='', remove_prefix=True):
  """Loads variables from checkpoint of policy trained by baselines."""

  # Forked from address below since we needed loading from different var names:
  # https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py
  variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
               if v.name.startswith(prefix)]

  loaded_params = joblib.load(load_path)
  restores = []
  for v in variables:
    v_name = v.name[len(prefix):] if remove_prefix else v.name
    restores.append(v.assign(loaded_params[v_name]))

  sess.run(restores)


class ObservationStacker(object):
  """Utility class that produces stacked observations."""

  def __init__(self, stacking):
    self._stacking = stacking
    self._data = []

  def get(self, observation):
    if self._data:
      self._data.append(observation)
      self._data = self._data[-self._stacking:]
    else:
      self._data = [observation] * self._stacking
    return np.concatenate(self._data, axis=-1)

  def reset(self):
    self._data = []


class DummyEnv(object):
  # We need env object to pass to build_policy, however real environment
  # is not there yet.

  def __init__(self, action_set, stacking):
    self.action_space = gym.spaces.Discrete(
        len(football_action_set.action_set_dict[action_set]))
    self.observation_space = gym.spaces.Box(
        0, 255, shape=[72, 96, 4 * stacking], dtype=np.uint8)
