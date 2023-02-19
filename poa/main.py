import argparse
from environment import Marketplace
import numpy as np
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

parser = argparse.ArgumentParser()
# TF params
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
parser.add_argument("--threads", default=10, type=int, help="Number of CPU threads to use.")
parser.add_argument("--seller_hidden_actor", default=32, type=int, help="Size of the hidden layer of the seller network.")
parser.add_argument("--seller_hidden_critic", default=256, type=int, help="Size of the hidden layer of the seller network.")
parser.add_argument("--buyer_hidden_actor", default=32, type=int, help="Size of the hidden layer of the buyer network.")
parser.add_argument("--buyer_hidden_critic", default=256, type=int, help="Size of the hidden layer of the buyer network.")
parser.add_argument("--actor_learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--critic_learning_rate", default=1e-3, type=float, help="Learning rate.")

# Simulation params
parser.add_argument("--num_sellers", default=10, type=int, help="Number of sellers on the market.")
parser.add_argument("--num_buyers", default=10, type=int, help="Number of buyers on the market.")
parser.add_argument("--batch_size", default=512, type=int, help="Batch size of the training.")
parser.add_argument("--l2", default=0.01, type=float, help="L2 regularization constant.")
parser.add_argument("--gamma", default=0.99, type=float, help="Decay factor.")
parser.add_argument("--c_mean", default=0.0, type=float, help="Decay factor.")
parser.add_argument("--clip_norm", default=0.1, type=float, help="Gradient clip norm.")
parser.add_argument("--target_tau", default=0.002, type=float, help="Target network update weight.")
parser.add_argument("--train_noise_var", default=0.05, type=float, help="Critic update.")
parser.add_argument("--train_noise_clip", default=0.0, type=float, help="Critic update.")
parser.add_argument("--episodes", default=3000, type=int, help="Training episodes.")
parser.add_argument("--eval_every", default=100, type=int, help="Evaluate every _ steps.")
parser.add_argument("--eval_for", default=100, type=int, help="Evaluation steps.")
parser.add_argument("--actor_update_freq", default=3, type=int, help="Train actor every _ critic trainings.")
parser.add_argument("--show_window", default=1_000_000, type=int, help="Train actor every _ critic trainings.")
parser.add_argument("--steps_in_episode", default=10, type=int, help="Number of days in each episode.")
parser.add_argument("--entropy_regularization_buyer", default=0.003, type=float, help="Variance of the folded gauss distribution to use initially.")
parser.add_argument("--entropy_regularization_seller", default=0.003, type=float, help="Variance of the folded gauss distribution to use initially.")

# Resources
parser.add_argument("--buyer_starting_money", default=0, type=float, help="The amount of money at the start of the simulation.")
parser.add_argument("--buyer_starting_supply", default=0, type=float, help="The amount of supply at the start of the simulation.")
parser.add_argument("--seller_starting_money", default=0, type=float, help="The amount of money at the start of the simulation.")
parser.add_argument("--seller_starting_supply", default=0, type=float, help="The amount of supply at the start of the simulation.")

parser.add_argument("--buyer_earning_per_day", default=1/8, type=float, help="The amount of money that the buyer recieves at the start of each day in the simulation.")
parser.add_argument("--seller_earning_per_day", default=1/4, type=float, help="The amount of supply that the seller recieves at the start of each day in the simulation.")
parser.add_argument("--demands", default=[1, 1, 1, 1, 1, 1, 5, 5, 5, 5], type=list, help="Good required per Market for agents.")
parser.add_argument("--earnings", default=[4, 4, 5, 5, 6, 6, 1, 1, 1, 1], type=list, help="Money gained by agents per step.")

parser.add_argument("--seller_resupply_model", default='constant', type=str, help="What amount of supply does a seller receive in a given day. Supported are: constant, rand_constant, cos, rand_cos.")
parser.add_argument("--sellers_share_buffer", default=False, type=bool, help="If the sellers randomly exchange samples used for training.")
parser.add_argument("--reward_shaping", default=False, type=bool, help="Use reward shaping.")
parser.add_argument("--shaping_const", default=0.1, type=float, help="Reward shaping scaling constant.")
parser.add_argument("--upgoing_policy", default=True, type=bool, help="Use upgoing policy update.")
parser.add_argument("--reward_clipping", default=True, type=bool, help="Clip rewards to [-clip_const,clip_const].")
parser.add_argument("--clip_const", default=1., type=float, help="Clipping constant.")
parser.add_argument("--fairness", default='talmud', type=str, help="Fairness models: free, talmud.")
parser.add_argument("--perishing_good", default=False, type=bool, help="Supplies are only stored by sellers for one step.")
parser.add_argument("--market_model", default='greedy', type=str, help="Market model to use: random, greedy, average, absolute.")

parser.add_argument("--consumption_on_step", default=1., type=float, help="Amount of supply consumed by buyer per step.")
parser.add_argument("--min_trade_volume", default=0., type=float, help="Minimum amount of supplies/rights to trade.")
parser.add_argument("--max_trade_volume", default=1., type=float, help="Maximum amount of money the rights/supplies can cost.")
parser.add_argument("--max_trade_price", default=1., type=float, help="Maximum amount of money the rights/supplies can cost.")
parser.add_argument("--renew_rights_every", default=1, type=int, help="Every _ steps to renew rights according to fairness.")
parser.add_argument("--in_stock_supply_reward", default=1., type=float, help="The reward gained by buyer if whole consumption is met.")
parser.add_argument("--missing_supply_reward", default=0., type=float, help="The reward gained by buyer if whole consumption is missing.")
parser.add_argument("--final_money_reward", default=1., type=float, help="The reward gained by buyer per unit of money at the end of an episode.")
parser.add_argument("--end_supply_reward", default=-1/8, type=float, help="The reward gained by seller for each unit of supply at the end of a step.")
parser.add_argument("--final_good_reward", default=1/2, type=float, help="The reward gained by seller for each unit of supply at the end of an episode.")

args = parser.parse_args([] if "__file__" not in globals() else None)

np.random.seed(args.seed)
tf.random.set_seed(args.seed)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

d, e = [], []
for n in range(args.num_buyers):
	if np.random.uniform() > 3/4:
		d.append(np.round(np.random.uniform(4, 6), 1))
		e.append(np.round(np.random.uniform(1, 2), 1))	
	else:
		d.append(np.round(np.random.uniform(1, 2), 1))
		e.append(np.round(np.random.uniform(4, 6), 1))

args.demands = d
args.earnings = e

path = os.getcwd()
dir_name = f'poa_{args.market_model}_{args.fairness}_{args.seed}_{args.num_buyers}'
os.mkdir(dir_name)
os.chdir(dir_name)
pickle.dump(args, open("args.pickle", "wb"))
os.mkdir('models')
os.chdir(path)

market = Marketplace(args, dir_name, multiagent=True, eval=False)
eval_market = Marketplace(args, dir_name, multiagent=True, eval=True)

def main():
	for batch in range(args.episodes // args.eval_every):
		print(f'Working on batch {batch + 1}.')
		for crisis in range(args.eval_every):
			market.simulate_crisis(args)
		market.save(batch, models=False)
		
main()
