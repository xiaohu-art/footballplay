import numpy as np
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#util
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == 'Dict':
        obs_shape = obs_space.spaces
    else:
        raise NotImplementedError
    return obs_shape
# module===============================================================
class MAPPOModule:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = PolicyNetwork(args, self.obs_space, self.act_space, self.device)
        #self.critic = ValueNetwork(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        #self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

# TODO get actions
    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        #values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks) # TODO critic???
        return None, actions, action_log_probs, rnn_states_actor, None # values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        action_log_probs, dist_entropy, policy_values = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, policy_values

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

# algo=================================================================
class MAPPOAlgorithm():
    def __init__(self,
                 args,
                 init_module,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.algo_module = init_module


        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.policy_value_loss_coef = args.policy_value_loss_coef
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_policy_vhead = args.use_policy_vhead

        self.cnt = 0
        
        #assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        #print("self._use_popart=",self._use_popart)x
        #print("self.value_normalizer",self.value_normalizer)
        #print("self.algo_module.critic.v_out",self.algo_module.critic.v_out)
        #print("self._use_valuenorm",self._use_valuenorm) o
        #print("self._use_policy_vhead",self._use_policy_vhead) x

        if self._use_valuenorm: 
            self.value_normalizer = ValueNorm(1, device = self.device)

    def prep_rollout(self):
        self.algo_module.actor.eval()
        #self.algo_module.critic.eval()

# network==============================================================
class PolicyNetwork(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal 
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks 
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N 
        self._use_policy_vhead = args.use_policy_vhead 
        self._recurrent_N = args.recurrent_N 
        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_shape=(268,)
        #obs_shape = get_shape_from_obs_space(obs_space)
        if 'Dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            self.base = CNNBase(args, obs_shape) if len(obs_shape)==3 else MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
            
        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size

        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)

        if self._use_policy_vhead:
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
            def init_(m): 
                return init(m, init_method, lambda x: nn.init.constant_(x, 0))
            if self._use_popart:
                self.v_out = init_(PopArt(input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(input_size, 1))

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):

        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        #raise ValueError('{} {} {}'.format(obs.shape, rnn_states.shape, masks.shape))
        
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        actor_features = self.base(obs) #

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        
        return actions, action_log_probs, rnn_states
    
    def get_log_1mp(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        # get log(1-p), source code copy from evaluate actions
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        actor_features = self.base(obs)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        action_log_probs = self.act.get_log_1mp(actor_features, action, available_actions, active_masks = active_masks if self._use_policy_active_masks else None)

        return action_log_probs

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        actor_features = self.base(obs)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions, active_masks = active_masks if self._use_policy_active_masks else None)

        values = self.v_out(actor_features) if self._use_policy_vhead else None
       
        return action_log_probs, dist_entropy, values

    def get_policy_values(self, obs, rnn_states, masks):        
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        
        values = self.v_out(actor_features)

        return values

class ValueNetwork(nn.Module):
    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super(ValueNetwork, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal  
        self._activation_id = args.activation_id     
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._use_popart = args.use_popart
        self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        if 'Dict' in share_obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            self.base = CNNBase(args, share_obs_shape) if len(share_obs_shape)==3 else MLPBase(args, share_obs_shape, use_attn_internal=True, use_cat_self=args.use_cat_self)

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(share_obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size

        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(input_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(input_size, 1))

        self.to(device)

class ValueNorm(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[:self.norm_axes])
            weight = self.beta ** batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        out = out.cpu().numpy()
        
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, activation_id, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=hidden_size//2, kernel_size=kernel_size, stride=stride), active_func,
            Flatten(),
            nn.Linear(hidden_size//2 * (input_width-kernel_size+stride) * (input_height-kernel_size+stride), hidden_size), active_func,
            nn.Linear(hidden_size, hidden_size), active_func)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        
        return x

class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._activation_id)

    def forward(self, x):
        x = self.cnn(x)
        return x

    @property
    def output_size(self):
        return self.hidden_size

#mlp
if True: #MLP
    
    class MLPLayer(nn.Module):
        def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, activation_id):
            super(MLPLayer, self).__init__()
            self._layer_N = layer_N

            active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
            gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            self.fc1 = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
            self.fc_h = nn.Sequential(init_(
                nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
            self.fc2 = get_clones(self.fc_h, self._layer_N)

        def forward(self, x):
            x = self.fc1(x)
            for i in range(self._layer_N):
                x = self.fc2[i](x)
            return x

    class CONVLayer(nn.Module):
        def __init__(self, input_dim, hidden_size, use_orthogonal, activation_id):
            super(CONVLayer, self).__init__()

            active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
            gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            self.conv = nn.Sequential(
                    init_(nn.Conv1d(in_channels=input_dim, out_channels=hidden_size//4, kernel_size=3, stride=2, padding=0)), active_func, #nn.BatchNorm1d(hidden_size//4),
                    init_(nn.Conv1d(in_channels=hidden_size//4, out_channels=hidden_size//2, kernel_size=3, stride=1, padding=1)), active_func, #nn.BatchNorm1d(hidden_size//2),
                    init_(nn.Conv1d(in_channels=hidden_size//2, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)), active_func) #, nn.BatchNorm1d(hidden_size))

        def forward(self, x):
            x = self.conv(x)
            return x

    class MLPBase(nn.Module):
        def __init__(self, args, obs_shape, use_attn_internal=False, use_cat_self=True):
            super(MLPBase, self).__init__()

            self._use_feature_normalization = args.use_feature_normalization
            self._use_orthogonal = args.use_orthogonal
            self._activation_id = args.activation_id
            self._use_attn = args.use_attn
            self._use_attn_internal = use_attn_internal
            self._use_average_pool = args.use_average_pool
            self._use_conv1d = args.use_conv1d
            self._stacked_frames = args.stacked_frames
            self._layer_N = 0 if args.use_single_network else args.layer_N
            self._attn_size = args.attn_size
            self.hidden_size = args.hidden_size

            obs_dim = obs_shape[0]

            if self._use_feature_normalization:
                self.feature_norm = nn.LayerNorm(obs_dim)

            if self._use_attn and self._use_attn_internal:
            
                if self._use_average_pool:
                    if use_cat_self:
                        inputs_dim = self._attn_size + obs_shape[-1][1]
                    else:
                        inputs_dim = self._attn_size
                else:
                    split_inputs_dim = 0
                    split_shape = obs_shape[1:]
                    for i in range(len(split_shape)):
                        split_inputs_dim += split_shape[i][0]
                    inputs_dim = split_inputs_dim * self._attn_size
                self.attn = Encoder(args, obs_shape, use_cat_self)
                self.attn_norm = nn.LayerNorm(inputs_dim)
            else:
                inputs_dim = obs_dim

            if self._use_conv1d:
                self.conv = CONVLayer(self._stacked_frames, self.hidden_size, self._use_orthogonal, self._activation_id)
                random_x = torch.FloatTensor(1, self._stacked_frames, inputs_dim//self._stacked_frames)
                random_out = self.conv(random_x)
                assert len(random_out.shape)==3
                inputs_dim = random_out.size(-1) * random_out.size(-2)

            self.mlp = MLPLayer(inputs_dim, self.hidden_size,
                                self._layer_N, self._use_orthogonal, self._activation_id)

        def forward(self, x):
            if self._use_feature_normalization:
                x = self.feature_norm(x)

            if self._use_attn and self._use_attn_internal:
                x = self.attn(x, self_idx=-1)
                x = self.attn_norm(x)

            if self._use_conv1d:
                batch_size = x.size(0)
                x = x.view(batch_size, self._stacked_frames, -1)
                x = self.conv(x)
                x = x.view(batch_size, -1)

            x = self.mlp(x)

            return x

        @property
        def output_size(self):
            return self.hidden_size

# enoder
if True:
    class Encoder(nn.Module):
        def __init__(self, args, split_shape, cat_self=True):
            super(Encoder, self).__init__()
            self._use_orthogonal = args.use_orthogonal
            self._activation_id = args.activation_id
            self._attn_N = args.attn_N
            self._attn_size = args.attn_size
            self._attn_heads = args.attn_heads
            self._dropout = args.dropout
            self._use_average_pool = args.use_average_pool
            self._cat_self = cat_self
            if self._cat_self:
                self.embedding = CatSelfEmbedding(
                    split_shape[1:], self._attn_size, self._use_orthogonal, self._activation_id)
            else:
                self.embedding = Embedding(
                    split_shape[1:], self._attn_size, self._use_orthogonal, self._activation_id)

            self.layers = get_clones(EncoderLayer(
                self._attn_size, self._attn_heads, self._dropout, self._use_orthogonal, self._activation_id), self._attn_N)
            self.norm = nn.LayerNorm(self._attn_size)

        def forward(self, x, self_idx=-1, mask=None):
            x, self_x = self.embedding(x, self_idx)
            for i in range(self._attn_N):
                x = self.layers[i](x, mask)
            x = self.norm(x)
            if self._use_average_pool:
                x = torch.transpose(x, 1, 2)
                x = F.avg_pool1d(x, kernel_size=x.size(-1)).view(x.size(0), -1)
                if self._cat_self:
                    x = torch.cat((x, self_x), dim=-1)
            x = x.view(x.size(0), -1)
            return x

    def split_obs(obs, split_shape):
        start_idx = 0
        split_obs = []
        for i in range(len(split_shape)):
            split_obs.append(
                obs[:, start_idx:(start_idx+split_shape[i][0]*split_shape[i][1])])
            start_idx += split_shape[i][0]*split_shape[i][1]
        return split_obs

    class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff=512, dropout=0.0, use_orthogonal=True, activation_id=1):
            super(FeedForward, self).__init__()
            # We set d_ff as a default to 2048
            active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
            gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            self.linear_1 = nn.Sequential(
                init_(nn.Linear(d_model, d_ff)), active_func, nn.LayerNorm(d_ff))

            self.dropout = nn.Dropout(dropout)
            self.linear_2 = init_(nn.Linear(d_ff, d_model))

        def forward(self, x):
            x = self.dropout(self.linear_1(x))
            x = self.linear_2(x)
            return x

    def ScaledDotProductAttention(q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

    class MultiHeadAttention(nn.Module):
        def __init__(self, heads, d_model, dropout=0.0, use_orthogonal=True):
            super(MultiHeadAttention, self).__init__()
            
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0))

            self.d_model = d_model
            self.d_k = d_model // heads
            self.h = heads

            self.q_linear = init_(nn.Linear(d_model, d_model))
            self.v_linear = init_(nn.Linear(d_model, d_model))
            self.k_linear = init_(nn.Linear(d_model, d_model))
            self.dropout = nn.Dropout(dropout)
            self.out = init_(nn.Linear(d_model, d_model))

        def forward(self, q, k, v, mask=None):

            bs = q.size(0)

            # perform linear operation and split into h heads

            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            # transpose to get dimensions bs * h * sl * d_model

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            # calculate attention
            scores = ScaledDotProductAttention(
                q, k, v, self.d_k, mask, self.dropout)

            # concatenate heads and put through final linear layer
            concat = scores.transpose(1, 2).contiguous()\
                .view(bs, -1, self.d_model)

            output = self.out(concat)

            return output

    class EncoderLayer(nn.Module):
        def __init__(self, d_model, heads, dropout=0.0, use_orthogonal=True, activation_id=False, d_ff=512, use_FF=False):
            super(EncoderLayer, self).__init__()
            self._use_FF = use_FF
            self.norm_1 = nn.LayerNorm(d_model)
            self.norm_2 = nn.LayerNorm(d_model)
            self.attn = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
            self.ff = FeedForward(d_model, d_ff, dropout, use_orthogonal, activation_id)
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)

        def forward(self, x, mask):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
            if self._use_FF:
                x2 = self.norm_2(x)
                x = x + self.dropout_2(self.ff(x2))
            return x

    class CatSelfEmbedding(nn.Module):
        def __init__(self, split_shape, d_model, use_orthogonal=True, activation_id=1):
            super(CatSelfEmbedding, self).__init__()
            self.split_shape = split_shape

            active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
            gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            for i in range(len(split_shape)):
                if i == (len(split_shape)-1):
                    setattr(self, 'fc_'+str(i), nn.Sequential(init_(
                        nn.Linear(split_shape[i][1], d_model)), active_func, nn.LayerNorm(d_model)))
                else:
                    setattr(self, 'fc_'+str(i), nn.Sequential(init_(nn.Linear(
                        split_shape[i][1]+split_shape[-1][1], d_model)), active_func, nn.LayerNorm(d_model)))

        def forward(self, x, self_idx=-1):
            x = split_obs(x, self.split_shape)
            N = len(x)

            x1 = []
            self_x = x[self_idx]
            for i in range(N-1):
                K = self.split_shape[i][0]
                L = self.split_shape[i][1]
                for j in range(K):
                    temp = torch.cat((x[i][:, (L*j):(L*j+L)], self_x), dim=-1)
                    exec('x1.append(self.fc_{}(temp))'.format(i))
            temp = x[self_idx]
            exec('x1.append(self.fc_{}(temp))'.format(N-1))

            out = torch.stack(x1, 1)

            return out, self_x

    class Embedding(nn.Module):
        def __init__(self, split_shape, d_model, use_orthogonal=True, activation_id=1):
            super(Embedding, self).__init__()
            self.split_shape = split_shape

            active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
            gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            for i in range(len(split_shape)):
                setattr(self, 'fc_'+str(i), nn.Sequential(init_(
                    nn.Linear(split_shape[i][1], d_model)), active_func, nn.LayerNorm(d_model)))

        def forward(self, x, self_idx=None):
            x = split_obs(x, self.split_shape)
            N = len(x)

            x1 = []
            for i in range(N):
                K = self.split_shape[i][0]
                L = self.split_shape[i][1]
                for j in range(K):
                    temp = x[i][:, (L*j):(L*j+L)]
                    exec('x1.append(self.fc_{}(temp))'.format(i))

            out = torch.stack(x1, 1)

            if self_idx is None:
                return out, None
            else:
                return out, x[self_idx]

#mix
if True:
    class Flatten(nn.Module):
        def forward(self, x):
            return x.reshape(x.size(0), -1)

    class MIXBase(nn.Module):
        def __init__(self, args, obs_shape, cnn_layers_params=None):
            super(MIXBase, self).__init__()

            self._use_orthogonal = args.use_orthogonal
            self._activation_id = args.activation_id
            self._use_maxpool2d = args.use_maxpool2d
            self.hidden_size = args.hidden_size
            self.cnn_keys = []
            self.embed_keys = []
            self.mlp_keys = []
            self.n_cnn_input = 0
            self.n_embed_input = 0
            self.n_mlp_input = 0

            for key in obs_shape:
                if obs_shape[key].__class__.__name__ == 'Box':
                    key_obs_shape = obs_shape[key].shape
                    if len(key_obs_shape) == 3:
                        self.cnn_keys.append(key)
                    else:
                        if "orientation" in key:
                            self.embed_keys.append(key)
                        else:
                            self.mlp_keys.append(key)
                else:
                    raise NotImplementedError

            if len(self.cnn_keys) > 0:
                self.cnn = self._build_cnn_model(obs_shape, cnn_layers_params, self.hidden_size, self._use_orthogonal, self._activation_id)
            if len(self.embed_keys) > 0:
                self.embed = self._build_embed_model(obs_shape)
            if len(self.mlp_keys) > 0:
                self.mlp = self._build_mlp_model(obs_shape, self.hidden_size, self._use_orthogonal, self._activation_id)

        def forward(self, x):
            out_x = x
            if len(self.cnn_keys) > 0:
                cnn_input = self._build_cnn_input(x)
                cnn_x = self.cnn(cnn_input)
                out_x = cnn_x

            if len(self.embed_keys) > 0:
                embed_input = self._build_embed_input(x)
                embed_x = self.embed(embed_input.long()).view(embed_input.size(0), -1)
                out_x = torch.cat([out_x, embed_x], dim=1)

            if len(self.mlp_keys) > 0:
                mlp_input = self._build_mlp_input(x)
                mlp_x = self.mlp(mlp_input).view(mlp_input.size(0), -1)
                out_x = torch.cat([out_x, mlp_x], dim=1) # ! wrong

            return out_x

        def _build_cnn_model(self, obs_shape, cnn_layers_params, hidden_size, use_orthogonal, activation_id):
            
            if cnn_layers_params is None:
                cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
            else:
                def _convert(params):
                    output = []
                    for l in params.split(' '):
                        output.append(tuple(map(int, l.split(','))))
                    return output
                cnn_layers_params = _convert(cnn_layers_params)
            
            active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
            gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            for key in self.cnn_keys:
                if key in ['rgb','depth','image','occupy_image']:
                    self.n_cnn_input += obs_shape[key].shape[2] 
                    cnn_dims = np.array(obs_shape[key].shape[:2], dtype=np.float32)
                elif key in ['global_map','local_map','global_obs','global_merge_obs','global_merge_goal','gt_map']:
                    self.n_cnn_input += obs_shape[key].shape[0] 
                    cnn_dims = np.array(obs_shape[key].shape[1:3], dtype=np.float32)
                else:
                    raise NotImplementedError

            cnn_layers = []
            prev_out_channels = None
            for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_layers_params):
                if self._use_maxpool2d and i != len(cnn_layers_params) - 1:
                    cnn_layers.append(nn.MaxPool2d(2))

                if i == 0:
                    in_channels = self.n_cnn_input
                else:
                    in_channels = prev_out_channels

                cnn_layers.append(init_(nn.Conv2d(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,)))
                #if i != len(cnn_layers_params) - 1:
                cnn_layers.append(active_func)
                prev_out_channels = out_channels

            for i, (_, kernel_size, stride, padding) in enumerate(cnn_layers_params):
                if self._use_maxpool2d and i != len(cnn_layers_params) - 1:
                    cnn_dims = self._maxpool_output_dim(dimension=cnn_dims,
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array([2, 2], dtype=np.float32),
                    stride=np.array([2, 2], dtype=np.float32))
                cnn_dims = self._cnn_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([padding, padding], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array([kernel_size, kernel_size], dtype=np.float32),
                    stride=np.array([stride, stride], dtype=np.float32),
                )
                
            cnn_layers += [
                Flatten(),
                init_(nn.Linear(cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1],
                            hidden_size)),
                active_func,
                nn.LayerNorm(hidden_size),
            ]
            return nn.Sequential(*cnn_layers)

        def _build_embed_model(self, obs_shape):
            self.embed_dim = 0
            for key in self.embed_keys:
                self.n_embed_input = 72
                self.n_embed_output = 8
                self.embed_dim += np.prod(obs_shape[key].shape)

            return nn.Embedding(self.n_embed_input, self.n_embed_output)

        def _build_mlp_model(self, obs_shape, hidden_size, use_orthogonal, activation_id):

            active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
            gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            for key in self.mlp_keys:
                self.n_mlp_input += np.prod(obs_shape[key].shape)

            return nn.Sequential(
                        init_(nn.Linear(self.n_mlp_input, hidden_size)), active_func, nn.LayerNorm(hidden_size))
            
        def _maxpool_output_dim(self, dimension, dilation, kernel_size, stride):
            """Calculates the output height and width based on the input
            height and width to the convolution layer.
            ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
            """
            assert len(dimension) == 2
            out_dimension = []
            for i in range(len(dimension)):
                out_dimension.append(
                    int(np.floor(
                        ((dimension[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                    ))
                )
            return tuple(out_dimension)

        def _cnn_output_dim(self, dimension, padding, dilation, kernel_size, stride):
            """Calculates the output height and width based on the input
            height and width to the convolution layer.
            ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
            """
            assert len(dimension) == 2
            out_dimension = []
            for i in range(len(dimension)):
                out_dimension.append(
                    int(np.floor(
                        ((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                    ))
                )
            return tuple(out_dimension)

        def _build_cnn_input(self, obs):
            cnn_input = []

            for key in self.cnn_keys:
                if key in ['rgb','depth','image','occupy_image']:
                    cnn_input.append(obs[key].permute(0, 3, 1, 2) / 255.0)
                elif key in ['global_map', 'local_map', 'global_obs', 'global_merge_obs', 'global_merge_goal','gt_map']:
                    cnn_input.append(obs[key])
                else:
                    raise NotImplementedError

            cnn_input = torch.cat(cnn_input, dim=1)
            return cnn_input

        def _build_embed_input(self, obs):
            embed_input = []
            for key in self.embed_keys:
                embed_input.append(obs[key].view(obs[key].size(0), -1))

            embed_input = torch.cat(embed_input, dim=1)
            return embed_input

        def _build_mlp_input(self, obs):
            mlp_input = []
            for key in self.mlp_keys:
                mlp_input.append(obs[key].view(obs[key].size(0), -1))

            mlp_input = torch.cat(mlp_input, dim=1)
            return mlp_input

        @property
        def output_size(self):
            output_size = 0
            if len(self.cnn_keys) > 0:
                output_size += self.hidden_size

            if len(self.embed_keys) > 0:
                output_size += 8 * self.embed_dim

            if len(self.mlp_keys) > 0:
                output_size += self.hidden_size
            return output_size

# rnn
if True:
    class RNNLayer(nn.Module):
        def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
            super(RNNLayer, self).__init__()
            self._recurrent_N = recurrent_N
            self._use_orthogonal = use_orthogonal

            self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    if self._use_orthogonal:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.xavier_uniform_(param)
            self.norm = nn.LayerNorm(outputs_dim)

        def forward(self, x, hxs, masks):
            #print(masks.__class__.__name__)
            if x.size(0) == hxs.size(0):
                x, hxs = self.rnn(x.unsqueeze(0), (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
                
                '''
                x1 = x.unsqueeze(0)
                masks2 = masks.repeat(1, self._recurrent_N)
                x2 = hxs * masks2.unsqueeze(-1)

                raise ValueError('=={}'.format(x2.shape))
                
                x2 = x2.transpose(0, 1).contiguous()

                x, hxs = self.rnn(x1, x2)
                '''
                x = x.squeeze(0)
                hxs = hxs.transpose(0, 1)
            else:
                N = hxs.size(0)
                T = int(x.size(0) / N)

                # unflatten
                x = x.view(T, N, x.size(1))

                # Same deal with masks
                masks = masks.view(T, N)

                # Let's figure out which steps in the sequence have a zero for any agent
                # We will always assume t=0 has a zero in it as that makes the logic cleaner
                has_zeros = ((masks[1:] == 0.0)
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

                # +1 to correct the masks[1:]
                if has_zeros.dim() == 0:
                    # Deal with scalar
                    has_zeros = [has_zeros.item() + 1]
                else:
                    has_zeros = (has_zeros + 1).numpy().tolist()

                # add t=0 and t=T to the list
                has_zeros = [0] + has_zeros + [T]

                hxs = hxs.transpose(0, 1)

                outputs = []
                for i in range(len(has_zeros) - 1):
                    # We can now process steps that don't have any zeros in masks together!
                    # This is much faster
                    start_idx = has_zeros[i]
                    end_idx = has_zeros[i + 1]               
                    temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                    rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                    outputs.append(rnn_scores)

                # assert len(outputs) == T
                # x is a (T, N, -1) tensor
                x = torch.cat(outputs, dim=0)

                # flatten
                x = x.reshape(T * N, -1)
                hxs = hxs.transpose(0, 1)

            x = self.norm(x)
            return x, hxs

#act
if True:
    class ACTLayer(nn.Module):
        def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
            super(ACTLayer, self).__init__()
            self.multidiscrete_action = False
            self.continuous_action = False
            self.mixed_action = False

            
            action_dim = 33
            #action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
            
        
        def forward(self, x, available_actions=None, deterministic=False):
            if self.mixed_action :
                actions = []
                action_log_probs = []
                for action_out in self.action_outs:
                    action_logit = action_out(x)
                    action = action_logit.mode() if deterministic else action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    actions.append(action.float())
                    action_log_probs.append(action_log_prob)

                actions = torch.cat(actions, -1)
                action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)

            elif self.multidiscrete_action:
                actions = []
                action_log_probs = []
                for action_out in self.action_outs:
                    action_logit = action_out(x)
                    action = action_logit.mode() if deterministic else action_logit.sample()
                    action_log_prob = action_logit.log_probs(action)
                    actions.append(action)
                    action_log_probs.append(action_log_prob)

                actions = torch.cat(actions, -1)
                action_log_probs = torch.cat(action_log_probs, -1)
            
            elif self.continuous_action:
                action_logits = self.action_out(x)
                actions = action_logits.mode() if deterministic else action_logits.sample() 
                action_log_probs = action_logits.log_probs(actions)
            
            else:
                action_logits = self.action_out(x, available_actions)
                actions = action_logits.mode() if deterministic else action_logits.sample() 
                action_log_probs = action_logits.log_probs(actions)
            
            return actions, action_log_probs

        def get_probs(self, x, available_actions=None):
            if self.mixed_action or self.multidiscrete_action:
                action_probs = []
                for action_out in self.action_outs:
                    action_logit = action_out(x)
                    action_prob = action_logit.probs
                    action_probs.append(action_prob)
                action_probs = torch.cat(action_probs, -1)
            elif self.continuous_action:
                action_logits = self.action_out(x)
                action_probs = action_logits.probs
            else:
                action_logits = self.action_out(x, available_actions)
                action_probs = action_logits.probs
            
            return action_probs

        def get_log_1mp(self, x, action, available_actions=None, active_masks=None):
            action_logits = self.action_out(x, available_actions)
            action_prob = torch.gather(action_logits.probs, 1, action.long())
            action_prob = torch.clamp(action_prob, 0, 1-1e-6)
            action_log_1mp = torch.log(1 - action_prob)
            return action_log_1mp

        def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
            if self.mixed_action:
                a, b = action.split((2, 1), -1)
                b = b.long()
                action = [a, b] 
                action_log_probs = [] 
                dist_entropy = []
                for action_out, act in zip(self.action_outs, action):
                    action_logit = action_out(x)
                    action_log_probs.append(action_logit.log_probs(act))
                    if active_masks is not None:
                        if len(action_logit.entropy().shape) == len(active_masks.shape):
                            dist_entropy.append((action_logit.entropy() * active_masks).sum()/active_masks.sum()) 
                        else:
                            dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum()/active_masks.sum())
                    else:
                        dist_entropy.append(action_logit.entropy().mean())
                    
                action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
                dist_entropy = dist_entropy[0] * 0.0025 + dist_entropy[1] * 0.01 

            elif self.multidiscrete_action:
                action = torch.transpose(action, 0, 1)
                action_log_probs = []
                dist_entropy = []
                for action_out, act in zip(self.action_outs, action):
                    action_logit = action_out(x)
                    action_log_probs.append(action_logit.log_probs(act))
                    if active_masks is not None:
                        dist_entropy.append((action_logit.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum())
                    else:
                        dist_entropy.append(action_logit.entropy().mean())

                action_log_probs = torch.cat(action_log_probs, -1) # ! could be wrong
                dist_entropy = torch.tensor(dist_entropy).mean()

            elif self.continuous_action:
                action_logits = self.action_out(x)
                action_log_probs = action_logits.log_probs(action)
                if active_masks is not None:
                    dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
                else:
                    dist_entropy = action_logits.entropy().mean()       
            else:
                action_logits = self.action_out(x, available_actions)
                action_log_probs = action_logits.log_probs(action)
                if active_masks is not None:
                    dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
                else:
                    dist_entropy = action_logits.entropy().mean()
            
            return action_log_probs, dist_entropy

# distributions
if True:
    # Categorical
    class FixedCategorical(torch.distributions.Categorical):
        def sample(self):
            return super().sample().unsqueeze(-1)

        def log_probs(self, actions):
            return (
                super()
                .log_prob(actions.squeeze(-1))
                .view(actions.size(0), -1)
                .sum(-1)
                .unsqueeze(-1)
            )

        def mode(self):
            return self.probs.argmax(dim=-1, keepdim=True)


    # Normal
    class FixedNormal(torch.distributions.Normal):
        def log_probs(self, actions):
            return super().log_prob(actions).sum(-1, keepdim=True)

        def entrop(self):
            return super.entropy().sum(-1)

        def mode(self):
            return self.mean


    # Bernoulli
    class FixedBernoulli(torch.distributions.Bernoulli):
        def log_probs(self, actions):
            return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

        def entropy(self):
            return super().entropy().sum(-1)

        def mode(self):
            return torch.gt(self.probs, 0.5).float()


    class Categorical(nn.Module):
        def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
            super(Categorical, self).__init__()
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
            def init_(m): 
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

            self.linear = init_(nn.Linear(num_inputs, num_outputs))

        def forward(self, x, available_actions=None):
            x = self.linear(x)
            if available_actions is not None:
                x[available_actions == 0] = -1e10
            return FixedCategorical(logits=x)


    class DiagGaussian(nn.Module):
        def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
            super(DiagGaussian, self).__init__()

            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
            def init_(m): 
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

            self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
            self.logstd = AddBias(torch.zeros(num_outputs))

        def forward(self, x):
            action_mean = self.fc_mean(x)

            #  An ugly hack for my KFAC implementation.
            zeros = torch.zeros(action_mean.size())
            if x.is_cuda:
                zeros = zeros.cuda()

            action_logstd = self.logstd(zeros)
            return FixedNormal(action_mean, action_logstd.exp())


    class Bernoulli(nn.Module):
        def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
            super(Bernoulli, self).__init__()
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
            def init_(m): 
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
            
            self.linear = init_(nn.Linear(num_inputs, num_outputs))

        def forward(self, x):
            x = self.linear(x)
            return FixedBernoulli(logits=x)

    class AddBias(nn.Module):
        def __init__(self, bias):
            super(AddBias, self).__init__()
            self._bias = nn.Parameter(bias.unsqueeze(1))

        def forward(self, x):
            if x.dim() == 2:
                bias = self._bias.t().view(1, -1)
            else:
                bias = self._bias.t().view(1, -1, 1, 1)

            return x + bias

# popart
if True:
    class PopArt(torch.nn.Module):
        
        def __init__(self, input_shape, output_shape, norm_axes=1, beta=0.99999, epsilon=1e-5, device=torch.device("cpu")):
            
            super(PopArt, self).__init__()

            self.beta = beta
            self.epsilon = epsilon
            self.norm_axes = norm_axes
            self.tpdv = dict(dtype=torch.float32, device=device)

            self.input_shape = input_shape
            self.output_shape = output_shape

            self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
            self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)
            
            self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
            self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
            self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
            self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

            self.reset_parameters()

        def reset_parameters(self):
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)
            self.mean.zero_()
            self.mean_sq.zero_()
            self.debiasing_term.zero_()

        def forward(self, input_vector):
            if type(input_vector) == np.ndarray:
                input_vector = torch.from_numpy(input_vector)
            input_vector = input_vector.to(**self.tpdv)

            return F.linear(input_vector, self.weight, self.bias)
        
        @torch.no_grad()
        def update(self, input_vector):
            if type(input_vector) == np.ndarray:
                input_vector = torch.from_numpy(input_vector)
            input_vector = input_vector.to(**self.tpdv)
            
            old_mean, old_stddev = self.mean, self.stddev

            batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
            batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

            self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
            self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
            self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

            self.stddev = (self.mean_sq - self.mean ** 2).sqrt().clamp(min=1e-4)

            self.weight = self.weight * old_stddev / self.stddev
            self.bias = (old_stddev * self.bias + old_mean - self.mean) / self.stddev

        def debiased_mean_var(self):
            debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
            debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(min=self.epsilon)
            debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
            return debiased_mean, debiased_var

        def normalize(self, input_vector):
            if type(input_vector) == np.ndarray:
                input_vector = torch.from_numpy(input_vector)
            input_vector = input_vector.to(**self.tpdv)

            mean, var = self.debiased_mean_var()
            out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
            
            return out

        def denormalize(self, input_vector):
            if type(input_vector) == np.ndarray:
                input_vector = torch.from_numpy(input_vector)
            input_vector = input_vector.to(**self.tpdv)

            mean, var = self.debiased_mean_var()
            out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
            
            out = out.cpu().numpy()

            return out