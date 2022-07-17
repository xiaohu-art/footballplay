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


"""Agent player controlled by the training policy and using step/reset API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from gfootball.env import player_base

# football_syw packages
import os
import random
import numpy as np
import torch
from pathlib import Path

from .algorithm import MAPPOAlgorithm as TrainAlgo
from .algorithm import MAPPOModule as AlgoModule
from .configs import get_config


class Player(player_base.PlayerBase):

  def __init__(self, player_config, env_config):
    player_base.PlayerBase.__init__(self, player_config)
    assert player_config['player_agent'] == 0, 'Only one \'agent\' player allowed'
    self._action = None

  def set_action(self, action):
    self._action = action

  def take_action(self, observations):
    return copy.deepcopy(self._action)


def _t2n(x):
    return x.detach().cpu().numpy()

class Driver(object):
    def __init__(self, config, client=None):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        self.actor_id = 0
        self.weight_ids = [0]

        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.learner_n_rollout_threads = self.all_args.learner_n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = not self.all_args.disable_wandb
        self.use_single_network = self.all_args.use_single_network
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval

        # reverb address
        self.program_type = self.all_args.program_type

        # dir
        self.model_dir = self.all_args.model_dir


        #share_observation_space = self.envs.share_observation_space[0] \
        #    if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.algo_module = AlgoModule(self.all_args, None, \
            None, None, device=self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.algo_module, device=self.device)
        self.model_keys = self.get_model_keys()
        self.init_clients(client)

    def init_clients(self, client=None):
        self.signal_client = None
        self.weight_client = None
        self.data_client = None

    def run(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    def collect_weight(self): #
        policy_actor = self.trainer.algo_module.actor
        #policy_critic = self.trainer.algo_module.critic
        model_weight = {'actor':policy_actor.state_dict()}#,'critic':policy_critic.state_dict()}
        return model_weight

    def get_model_keys(self):
        model_weight = self.collect_weight()
        model_keys = []
        for model_key in model_weight:
            keys = model_weight[model_key].keys()
            for key in keys:
                model_keys.append(model_key + '@' + key)
        return model_keys

    def restore(self):# 读模型
        policy_actor_state_dict = torch.load(os.path.dirname(os.path.abspath(__file__)) + '/actor/actor_buildin_30.pt', map_location=self.device) #TODO
        self.algo_module.actor.load_state_dict(policy_actor_state_dict)
    
# football driver
class FootballDriver(Driver):
    def __init__(self, config):
        super(FootballDriver, self).__init__(config)

#base_runner
class Runner:
    def __init__(self, argv):
        self.argv = argv
    def run(self):
        raise NotImplementedError

#base trainer
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer(Runner):
    def __init__(self, argv, program_type='local', client=None): #?
        super().__init__(argv)
        parser = get_config()
        all_args = self.extra_args_func(argv, parser)
        self.algorithm_name = all_args.algorithm_name
        all_args.program_type = program_type
        # reverb server
        #server = Server(all_args)
        set_seed(all_args.seed)
        # deal with multi-actors
        all_args.learner_n_rollout_threads = all_args.n_rollout_threads

        # cuda
        # device
        device = torch.device("cpu") #TODO
        #device = torch.device("cuda")

        # env init 
        #Env_Class, SubprocVecEnv, DummyVecEnv = self.get_env()
        #envs, eval_envs = self.env_init(all_args, Env_Class, SubprocVecEnv, DummyVecEnv)
        num_agents = all_args.num_agents

        config = {
            "all_args": all_args,
            "envs": None,
            "eval_envs": None,
            "num_agents": num_agents,
            "device": device
        }
        self.all_args, self.envs, self.eval_envs, self.config, self.server = \
            all_args, None, None, config, None
        self.driver = self.init_driver()

    def run(self): 
        self.driver.run()
        self.stop()

    def extra_args_func(self, argv, parser):
        raise NotImplementedError

    def get_env(self):
        raise NotImplementedError

    def init_driver(self):
        raise NotImplementedError

    def stop(self):
        self.envs.close()
        if self.all_args.use_eval and self.eval_envs is not self.envs:
            self.eval_envs.close()
        self.server.stop()
    
# football trainer
class FootballTrainer(Trainer):
    def __init__(self, argv):
        super(FootballTrainer, self).__init__(argv)

    def extra_args_func(self, args, parser): #
        #TODO：
        all_args = parser.parse_args()
        return all_args

    def init_driver(self): #
        driver = Driver(self.config)
        return driver



