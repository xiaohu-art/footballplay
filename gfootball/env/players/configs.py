import argparse

def get_config():

    parser = argparse.ArgumentParser(description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)
        
    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,default='rmappo')
    parser.add_argument("--experiment_name", type=str, default="tikick_v1",
                        help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for numpy/torch")
    parser.add_argument("--disable_cuda", action='store_true', default=True,
                        help="by default False, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for training rollout")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollout")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='shiyu',
                        help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--wandb_entity", type=str, default='tmarl',
                        help="[for wandb usage], to specify entity for simply collecting training data.")
    parser.add_argument("--disable_wandb", action='store_true', default=False,
                        help="[for wandb usage], by default False, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='gfootball',
                        help="specify the name of environment")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Dimension of hidden layers for actor/critic networks")  # TODO @zoeyuchao. The same comment might in need of change.
    parser.add_argument("--layer_N", type=int, default=3,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--activation_id", type=int,
                        default=1, help="choose 0 to use tanh, 1 to use relu, 2 to use leaky relu, 3 to use elu")
    parser.add_argument("--use_popart", action='store_true', default=False,
                        help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True,
                        help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    parser.add_argument("--cnn_layers_params", type=str, default=None,
                        help="The parameters of cnn layer")
    parser.add_argument("--use_maxpool2d", action='store_true',
                        default=False, help="Whether to apply layernorm to the inputs")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1,
                        help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=25,
                        help="Time length of chunks used to train a recurrent_policy")
    parser.add_argument("--use_influence_policy", action='store_true',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--influence_layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")

    # attn parameters
    parser.add_argument("--use_attn", action='store_true', default=False,
                        help=" by default False, use attention tactics.")
    parser.add_argument("--attn_N", type=int, default=1,
                        help="the number of attn layers, by default 1")
    parser.add_argument("--attn_size", type=int, default=64,
                        help="by default, the hidden size of attn layer")
    parser.add_argument("--attn_heads", type=int, default=4,
                        help="by default, the # of multiply heads")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="by default 0, the dropout ratio of attn layer.")
    parser.add_argument("--use_average_pool",
                        action='store_false', default=True, help="by default True, use average pooling for attn model.")
    parser.add_argument("--use_attn_internal", action='store_false', default=True,
                        help="by default True, whether to strengthen own characteristics")
    parser.add_argument("--use_cat_self", action='store_false', default=True,
                        help="by default True, whether to strengthen own characteristics")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--tau", type=float, default=0.995,
                        help='soft update polyak (default: 0.995)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=10,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_policy_vhead",
                        action='store_true', default=False,
                        help="by default, do not use policy vhead. if set, use policy vhead.")
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True,
                        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--policy_value_loss_coef", type=float,
                        default=1, help='policy value loss coefficient (default: 0.5)')
    parser.add_argument("--entropy_coef", type=float, default=0.00,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True,
                        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.999,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.995,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True,
                        help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float,
                        default=10.0, help=" coefficience of huber loss.")

    # ppg parameters
    parser.add_argument("--use_single_network", action='store_true',
                        default=False, help="Whether to use centralized V function")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=True,
                        help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=64,
                        help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--use_render", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default='.',
                        help="by default None. set the path to pretrained model.")

    parser.add_argument("--run_dir", type=str, default= None,
                        help="root dir to save curves, logs and models.")

    # distributed parameters
    parser.add_argument("--program_type", type=str, default="local",
                        help="running type of current program.", choices=["local", "whole", "actor", "learner", "server", "server_learner"])
    '''
    # replay buffer parameters
    parser.add_argument('--use_reward_normalization', action='store_true',
                        default=False, help="Whether to normalize rewards in replay buffer")
    parser.add_argument('--buffer_size', type=int, default=5000,
                        help="Max # of transitions that replay buffer can contain")
    parser.add_argument('--popart_update_interval_step', type=int, default=2,
                        help="After how many train steps popart should be updated")
    '''
    parser.add_argument('--scenario_name', type=str,
                            default='11_vs_11_kaggle', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int, default=0, help="number of players")
    parser.add_argument('--num_enemys', type=int, default=0, help="number of enemys")
    # football config
    parser.add_argument('--representation', type=str,
                        default='raw', help="format of the observation in gfootball env")
    parser.add_argument('--rewards', type=str,
                        default='scoring', help="format of the reward in gfootball env")
    parser.add_argument("--render_only", action='store_true', default=False,
                        help="if ture, render without training")

    parser.add_argument('--players', type=str,
                        default='raw', help="format of the observation in gfootball env")
    
    return parser