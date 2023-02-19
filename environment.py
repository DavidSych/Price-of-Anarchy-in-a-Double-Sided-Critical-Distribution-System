import numpy as np
import tensorflow as tf
import os
import networkx as nx
from mip import *

from agents import Seller, Buyer


def talmud_split(needs, total_supply):
	if np.sum(needs) / 2 == total_supply:
		return needs / 2
	
	elif np.sum(needs) / 2 > total_supply:  # not even half way full
		previous_demand = 0
		claims = np.zeros(len(needs))
		sorted_indexes = np.argsort(needs)
		for i, demand in enumerate(needs[sorted_indexes]):
			if total_supply - (len(needs) - i) * (
					demand - previous_demand) / 2 > 0:  # something is left over after this step
				total_supply -= (len(needs) - i) * (demand - previous_demand) / 2
				claims[i:] += (demand - previous_demand) / 2
				previous_demand = demand
			else:  # there is not enough liquid to bet to the top of the next cylinder
				claims[i:] += total_supply / (len(needs) - i)
				break
		
		out = np.zeros(len(needs))
		out[sorted_indexes] = claims
		return out
	
	else:  # more than half is full
		previous_demand = 0
		losses = np.zeros(len(needs))
		sorted_indexes = np.argsort(needs)
		total_loss = np.sum(needs) - total_supply
		for i, demand in enumerate(np.array(needs)[sorted_indexes]):
			if total_loss - (len(needs) - i) * (
					demand - previous_demand) / 2 > 0:  # something is left over after this step
				total_loss -= (len(needs) - i) * (demand - previous_demand) / 2
				losses[i:] += (demand - previous_demand) / 2
				previous_demand = demand
			else:  # there is not enough liquid to get to the top of the next cylinder
				losses[i:] += total_loss / (len(needs) - i)
				break
		
		out = np.zeros(len(needs))
		out[sorted_indexes] = losses
		return needs - out


class Marketplace():
	def __init__(self, args, dir_name, multiagent, eval):
		self.multiagent = multiagent
		self.eval = eval
		self.dir_name = dir_name
		self.args = args
		self.sellers = [Seller(args, s) for s in range(args.num_sellers)]
		self.buyers = [Buyer(args, b) for b in range(args.num_buyers)]
		
		self.demands = args.demands / np.mean(args.demands)
		self.earnings = args.earnings / np.mean(args.earnings)
		
		self.ep_num, self.step_num = 0, 0
		
		# SELLER SHAPE
		# (0) Market num [0,1], (1) seller Good, (2) offered volume percentage [0,1],
		# (3) offered price percentage [0,1], (4) reward, (5) Good after trading
		
		# BUYER SHAPE
		# (0) Market num [0,1], (1) Good, (2) Money, (3) Right,
		# (4) offered volume percentage [0,1], (5) offered price percentage [0,1] of Right
		# (6) desired volume percentage [0,1], (7) desired price percentage [0,1] of Good
		# (8) desired volume percentage [0,1], (9) desired price percentage [0,1] of Right
		# (10) reward, (11) Good, (12) Money, (13) Right after trading
		
		if multiagent and eval:
			seller_shape = (args.eval_for, args.steps_in_episode, args.num_sellers, 6)
			buyer_shape = (args.eval_for, args.steps_in_episode, args.num_buyers, 14)
		elif multiagent:
			seller_shape = (args.episodes, args.steps_in_episode, args.num_sellers, 6)
			buyer_shape = (args.episodes, args.steps_in_episode, args.num_buyers, 14)
		else:
			seller_shape = (args.eval_for + args.episodes, args.steps_in_episode, args.num_sellers, 6)
			buyer_shape = (args.eval_for + args.episodes, args.steps_in_episode, args.num_buyers, 14)
		
		self.seller_states = np.zeros(shape=seller_shape, dtype=np.float32)
		self.buyer_states = np.zeros(shape=buyer_shape, dtype=np.float32)
		
	def offered_volume_good(self):
		return self.seller_states[self.ep_num, self.step_num, :, 1] * self.seller_states[self.ep_num, self.step_num, :, 2]
	
	def scaled_seller_state(self):
		seller_states = self.seller_states[self.ep_num, self.step_num, :, :5].copy()
		seller_states[:, 2] = seller_states[:, 2] * seller_states[:, 1]
		seller_states[:, 3] = seller_states[:, 3] * self.args.max_trade_price
		return seller_states
	
	def scaled_seller_states(self):
		seller_states = self.seller_states.copy()
		seller_states[..., 2] = seller_states[..., 2] * seller_states[..., 1]
		seller_states[..., 3] = seller_states[..., 3] * self.args.max_trade_price
		return seller_states
	
	def scaled_buyer_state(self):
		buyer_states = self.buyer_states[self.ep_num, self.step_num, :, :11].copy()
		buyer_states[:, 4] = buyer_states[:, 4] * buyer_states[:, 3]
		buyer_states[:, 5] *= self.args.max_trade_price
		buyer_states[:, 6] *= self.args.max_trade_volume
		buyer_states[:, 7] *= self.args.max_trade_price
		buyer_states[:, 8] *= self.args.max_trade_volume
		buyer_states[:, 9] *= self.args.max_trade_price
		return buyer_states
	
	def scaled_buyer_states(self):
		buyer_states = self.buyer_states.copy()
		buyer_states[..., 4] = buyer_states[..., 4] * buyer_states[..., 3]
		buyer_states[..., 5] *= self.args.max_trade_price
		buyer_states[..., 6] *= self.args.max_trade_volume
		buyer_states[..., 7] *= self.args.max_trade_price
		buyer_states[..., 8] *= self.args.max_trade_volume
		buyer_states[..., 9] *= self.args.max_trade_price
		return buyer_states
	
	def resupply(self, args):
		# Give new Good to sellers
		if args.seller_resupply_model == 'constant':
			new_good = args.seller_earning_per_day * np.ones(args.num_sellers)
		elif args.seller_resupply_model == 'rand_constant':
			new_good = np.maximum(np.random.normal(args.seller_earning_per_day, args.seller_earning_per_day / 10), 0)
		else:
			raise Exception('Unknown resupply model.')
		
		if args.perishing_good or self.step_num == 0:
			self.seller_states[self.ep_num, self.step_num, :, 1] = new_good
		else:
			self.seller_states[self.ep_num, self.step_num, :, 1] += new_good
		
		# Give buyers more Money
		if self.step_num == 0:
			self.buyer_states[self.ep_num, self.step_num, :, 2] = args.buyer_earning_per_day * self.earnings
		else:
			self.buyer_states[self.ep_num, self.step_num, :, 2] += args.buyer_earning_per_day * self.earnings
	
	@tf.function
	def offer_good_tf(self, seller_states, buyer_states):
		actions = tf.TensorArray(tf.float32, size=self.args.num_sellers)
		for i, seller in enumerate(self.sellers):
			action = seller.action(seller_states[tf.newaxis, i, ...], buyer_states[tf.newaxis, ...])[0]
			
			actions = actions.write(i, action)
		
		actions = actions.stack()
		return actions
	
	def offer_good(self):
		seller_states = self.buyer_states[self.ep_num, self.step_num, :, 1:2].copy()
		buyer_states = self.buyer_states[self.ep_num, self.step_num, :, 1:3].copy()
		
		actions = self.offer_good_tf(seller_states, buyer_states)
		
		self.seller_states[self.ep_num, self.step_num, :, 2:4] = actions
	
	def distribute_right(self):
		offered_volume = np.sum(self.offered_volume_good())
		need = np.maximum(self.demands - self.buyer_states[self.ep_num, self.step_num, :, 1], 0)
		new_rights = talmud_split(need, offered_volume)
		self.buyer_states[self.ep_num, self.step_num, :, 3] = new_rights
	
	@tf.function
	def offer_right_tf(self, seller_states, buyer_states):
		actions = tf.TensorArray(tf.float32, size=self.args.num_buyers)
		for i, buyer in enumerate(self.buyers):
			action = buyer.sell_action(seller_states[tf.newaxis, ...], buyer_states[tf.newaxis, ...])[0]
			
			actions = actions.write(i, action)
		
		actions = actions.stack()
		return actions
	
	def offer_right(self):
		seller_states = self.scaled_seller_state()[:, 2:4]
		buyer_states = self.buyer_states[self.ep_num, self.step_num, :, 1:4].copy()
		
		actions = self.offer_right_tf(seller_states, buyer_states)
		
		self.buyer_states[self.ep_num, self.step_num, :, 4:6] = actions
	
	@tf.function
	def make_orders_tf(self, seller_states, buyer_states):
		actions = tf.TensorArray(tf.float32, size=self.args.num_buyers)
		for i, buyer in enumerate(self.buyers):
			action = buyer.buy_action(seller_states[tf.newaxis, ...], buyer_states[tf.newaxis, ...])[0]
			
			actions = actions.write(i, action)
		
		actions = actions.stack()
		return actions
	
	def make_orders(self):
		seller_states = self.scaled_seller_state()[:, 2:4]
		buyer_states = self.scaled_buyer_state()[:, 1:6]
		
		actions = self.make_orders_tf(seller_states, buyer_states)
		
		self.buyer_states[self.ep_num, self.step_num, :, 6:10] = actions
	
	def greedy_market_mechanism(self, args):
		seller_states = self.scaled_seller_state()[:, :4]
		buyer_states = self.scaled_buyer_state()[:, :10]
		
		vol_good, prc_good = seller_states[:, 2], seller_states[:, 3]
		vol_right, prc_right = buyer_states[:, 4], buyer_states[:, 5]
		des_vol_good, des_prc_good = buyer_states[:, 6], buyer_states[:, 7]
		des_vol_right, des_prc_right = buyer_states[:, 8], buyer_states[:, 9]
		available_right = buyer_states[:, 3] - vol_right
		available_money = buyer_states[:, 2]
		
		if args.market_model == 'greedy':
			buying_order = np.argsort(des_prc_good)[::-1]
			seller_order = np.argsort(prc_good)
			right_seller_order = np.argsort(prc_right)
		else:
			buying_order = np.random.permutation(args.num_buyers)
			seller_order = np.random.permutation(args.num_sellers)
			right_seller_order = np.random.permutation(args.num_buyers)
		
		g = np.zeros(shape=(args.num_sellers, args.num_buyers))
		r = np.zeros(shape=(args.num_buyers, args.num_buyers))
		
		for b in buying_order:
			# First, buy whatever you can with available Right
			for s in seller_order:
				if prc_good[s] > des_prc_good[b]:
					continue
				
				right_vol = available_right[b] if args.fairness[-6:] == 'talmud' else args.max_trade_volume
				affordable_vol = available_money[b] / prc_good[s] if prc_good[s] > 0 else args.max_trade_volume
				
				vol_to_buy = min(right_vol, affordable_vol, vol_good[s], des_vol_good[b]) - 1e-4
				if vol_to_buy > 0:
					g[s, b] += vol_to_buy
					vol_good[s] -= vol_to_buy
					des_vol_good[b] -= vol_to_buy
					available_money[b] -= vol_to_buy * prc_good[s]
					available_right[b] -= vol_to_buy
			
			# Second, buy Right and Good in pairs starting with the cheapest offer
			for s in seller_order:
				for t in right_seller_order:
					if prc_good[s] > des_prc_good[b] or prc_right[t] > des_prc_right[b] or t == b:
						continue
					
					affordable_vol = available_money[b] / (prc_good[s] + prc_right[t]) if prc_good[s] + prc_right[
						t] > 0 else args.max_trade_volume
					
					vol_to_buy = min(affordable_vol, vol_good[s], vol_right[t], des_vol_good[b],
									 des_vol_right[b]) - 1e-4
					if vol_to_buy > 0:
						g[s, b] += vol_to_buy
						r[t, b] += vol_to_buy
						vol_good[s] -= vol_to_buy
						vol_right[t] -= vol_to_buy
						des_vol_good[b] -= vol_to_buy
						des_vol_right[b] -= vol_to_buy
						available_money[b] -= vol_to_buy * (prc_good[s] + prc_right[t])
		
		e, s = self.ep_num, self.step_num
		
		# Copy whatever I had last market
		self.seller_states[e, s, :, 5:6] = self.seller_states[e, s, :, 1:2].copy()
		self.buyer_states[e, s, :, 11:14] = self.buyer_states[e, s, :, 1:4].copy()

		# Good
		self.seller_states[e, s, :, 5] -= np.sum(g, axis=1)
		self.buyer_states[e, s, :, 11] += np.sum(g, axis=0)
		
		# Money
		cost_right = np.sum((r.T * prc_right).T, axis=0)
		cost_supply = np.sum((g.T * prc_good).T, axis=0)
		cost = cost_right + cost_supply
		earning = np.sum(r, axis=1) * prc_right
		self.buyer_states[e, s, :, 12] += earning - cost
		
		# Right
		next_right = available_right + vol_right - np.sum(r, axis=1)
		self.buyer_states[e, s, :, 13] = next_right
	
	def max_flow_market_mechanism(self, args):
		seller_states = self.scaled_seller_state()[:, :4]
		buyer_states = self.scaled_buyer_state()[:, :10]
		
		vol_good, prc_good = seller_states[:, 2], seller_states[:, 3]
		vol_right, prc_right = buyer_states[:, 4], buyer_states[:, 5]
		des_vol_good, des_prc_good = buyer_states[:, 6], buyer_states[:, 7]
		des_vol_right, des_prc_right = buyer_states[:, 8], buyer_states[:, 9]
		available_right = buyer_states[:, 3] - vol_right
		available_money = buyer_states[:, 2]
		
		verbose = 0
		epsilon = 1e-4
		
		model = Model(sense=MAXIMIZE, solver_name=CBC)
		
		model.threads = self.args.threads
		model.verbose = 0
		
		## Variables
		
		S = range(self.args.num_sellers)
		T = range(self.args.num_buyers)
		B = range(self.args.num_buyers)
		
		#   If the buyer b is buying i items b_i = 1, 0 otherwise.
		g = np.array([[model.add_var(var_type=CONTINUOUS, name=(f"g[{s},{b}]")) for b in B] for s in S])
		r = np.array([[model.add_var(var_type=CONTINUOUS, name=(f"r[{t},{b}]")) for b in B] for t in T])
		
		min = model.add_var(var_type=CONTINUOUS, name=("min"))
		max = model.add_var(var_type=CONTINUOUS, name=("max"))
		
		## Objective
		
		upper_bound = 1000
		model.objective = upper_bound * xsum([xsum([g[s][b] for s in S]) for b in B])# + min - max
		
		## Constraints
		
		# We assume no conflict between buyer, seller and trader.
		for b in B:
			for s in S:
				# Positivity.
				model += g[s][b] >= 0
			for t in T:
				# Positivity
				model += r[t][b] >= 0
		
		for b in B:
			model += xsum(g[s][b] for s in S) >= min
			model += xsum(g[s][b] for s in S) <= max
			
			# Not above demand.
			model += xsum(g[s][b] for s in S) <= des_vol_good[b]
			model += xsum(r[t][b] for t in T) <= des_vol_right[b]
			
			# Affordability constraint
			model += xsum(g[s][b] * prc_good[s] for s in S) + xsum(r[t][b] * prc_right[t] for t in T) <= np.clip(available_money[b] - epsilon, 0, np.inf)
			
			# As much right as good.
			if self.args.fairness[-6:] == 'talmud':
				model += xsum(g[s][b] for s in S) <= xsum(r[t][b] for t in T) + np.clip(available_right[b] - epsilon, 0, np.inf)
			
			# No self-trading
			model += r[b][b] == 0
			
			# Constrain average price
			if args.market_model == 'average':
				model += xsum(g[s][b] * prc_good[s] for s in S) <= des_prc_good[b] * xsum(g[s][b] for s in S)
				model += xsum(r[t][b] * prc_right[t] for t in T) <= des_prc_right[b] * xsum(r[t][b] for t in T)
			else:
				for s in S:
					if prc_good[s] > 0:
						model += g[s][b] * prc_good[s] <= des_prc_good[b] * g[s][b]
				for t in T:
					if prc_right[t] > 0:
						model += r[t][b] * prc_right[t] <= des_prc_right[b] * r[t][b]
		
			# Higher percentage constraint
			#for t in T:
			#	if des_prc_good[b] > des_prc_good[t]:
			#		model += xsum(g[s][b] for s in S) / self.demands[b] >= xsum(g[s][t] for s in S) / self.demands[t]


		for s in S:
			# seller supply
			model += xsum(g[s][b] for b in B) <= np.clip(vol_good[s] - epsilon, 0, np.inf)
		
		for t in T:
			# trader supply
			model += xsum(r[t][b] for b in B) <= np.clip(vol_right[t] - epsilon, 0, np.inf)
		
		status = model.optimize()
		
		if verbose > 1:
			for c in model.constrs:
				if c.slack < 1 and "once" not in c.name and "all supply sold" not in c.name:
					print(c.name, c.slack)
		
		total = 0
		
		matrix_g = np.zeros((args.num_sellers, args.num_buyers))
		matrix_r = np.zeros((args.num_buyers, args.num_buyers))
		
		if status == OptimizationStatus.OPTIMAL:
			for s in S:
				for b in B:
					if g[s][b].x > 0:
						if verbose > 0:
							print(f"{g[s][b]}: {g[s][b].x}.")
						total += g[s][b].x
					matrix_g[s][b] = g[s][b].x
			for t in T:
				for b in B:
					if r[t][b].x > 0:
						if verbose > 0:
							print(f"{r[t][b]}: {r[t][b].x}.")
					matrix_r[t][b] = r[t][b].x
			if verbose > 0:
				print(f"total: {total}")
		else:
			print("model infeasible.")
				
		e, s = self.ep_num, self.step_num
		
		# Copy whatever I had last market
		self.seller_states[e, s, :, 5:6] = self.seller_states[e, s, :, 1:2].copy()
		self.buyer_states[e, s, :, 11:14] = self.buyer_states[e, s, :, 1:4].copy()
		
		# Good
		self.seller_states[e, s, :, 5] -= np.sum(matrix_g, axis=1)
		self.buyer_states[e, s, :, 11] += np.sum(matrix_g, axis=0)
		
		# Money
		cost_right = np.sum((matrix_r.T * prc_right).T, axis=0)
		cost_supply = np.sum((matrix_g.T * prc_good).T, axis=0)
		cost = cost_right + cost_supply
		earning = np.sum(matrix_r, axis=1) * prc_right
		self.buyer_states[e, s, :, 12] += earning - cost
		
		# Right
		next_right = available_right + vol_right - np.sum(matrix_r, axis=1)
		self.buyer_states[e, s, :, 13] = next_right

		self.seller_states[e, s, :, 5] = np.clip(self.seller_states[e, s, :, 5], 0, self.seller_states[e, s, :, 1])
		self.buyer_states[e, s, :, 11] = np.clip(self.buyer_states[e, s, :, 11], 0, np.inf)
		self.buyer_states[e, s, :, 12] = np.clip(self.buyer_states[e, s, :, 12], 0, np.inf)
		self.buyer_states[e, s, :, 13] = np.clip(self.buyer_states[e, s, :, 13], 0, self.buyer_states[e, s, :, 3])
	
	def trade(self, args):
		if args.market_model == 'greedy' or args.market_model == 'random':
			self.greedy_market_mechanism(args)
		elif args.market_model == 'average' or args.market_model == 'absolute':
			self.max_flow_market_mechanism(args)
		else:
			raise Exception('Unknown market mechanism.')
	
	def reward_agents(self, args):
		e, s = self.ep_num, self.step_num
		
		sold_good = self.seller_states[e, s, :, 1] - self.seller_states[e, s, :, 5]
		reward = sold_good * self.seller_states[e, s, :, 3] * args.max_trade_price
		self.seller_states[e, s, :, 4] = reward
		
		good = self.buyer_states[e, s, :, 11].copy()
		reward = np.minimum(good, self.demands / args.renew_rights_every)
		self.buyer_states[e, s, :, 10] = reward
		
		if args.reward_shaping and not self.eval:
			money_now = self.buyer_states[e, s, :, 2]
			money_next = self.buyer_states[e, s, :, 12] + self.earnings
			shaping_reward = money_now - args.gamma * money_next
			self.buyer_states[e, s, :, 10] += args.shaping_const * shaping_reward / self.earnings
			
			good_now = self.seller_states[e, s, :, 1]
			good_next = self.seller_states[e, s, :, 5]
			shaping_reward = good_now - args.gamma * good_next
			self.seller_states[e, s, :, 4] += args.shaping_const * shaping_reward
	
	def transfer_to_next_trade(self, args):
		if self.step_num + 1 == args.steps_in_episode:
			return
		
		e, s, _s = self.ep_num, self.step_num, self.step_num + 1
		good = self.buyer_states[e, s, :, 11].copy()
		new_good = np.maximum(good - self.demands / args.renew_rights_every, 0)
		self.buyer_states[e, _s, :, 1] = new_good
		self.buyer_states[e, _s, :, 2] = self.buyer_states[e, s, :, 12].copy()
		self.buyer_states[e, _s, :, 3] = self.buyer_states[e, s, :, 13].copy()
		
		next_good = self.seller_states[e, s, :, 5].copy()
		self.seller_states[e, _s, :, 1] = next_good
		prev_offered_vol = self.seller_states[e, s, :, 1] * self.seller_states[e, s, :, 2]
		sold_vol = self.seller_states[e, s, :, 2] - next_good
		self.seller_states[e, _s, :, 2] = (prev_offered_vol - sold_vol) / next_good
		self.seller_states[e, _s, np.where(next_good == 0), 2] = 0
		self.seller_states[e, _s, :, 3] = self.seller_states[e, s, :, 3]

	@tf.function
	def train_single_seller_actor(self, s_s, n_s_s, b_s, n_b_s, chosen_action, to_train):
		self.sellers[to_train].train_actor(s_s, n_s_s, b_s, n_b_s, chosen_action)

	@tf.function
	def train_single_seller_critic(self, s_s, n_s_s, b_s, n_b_s, to_train):
		self.sellers[to_train].train_critic(s_s, n_s_s, b_s, n_b_s)

	@tf.function
	def train_single_buyer_actor(self, s_s, n_s_s, b_s, n_b_s, chosen_action, to_train):
		self.buyers[to_train - self.args.num_sellers].train_actor(s_s, n_s_s, b_s, n_b_s, chosen_action)

	@tf.function
	def train_single_buyer_critic(self, s_s, n_s_s, b_s, n_b_s, to_train):
		self.buyers[to_train - self.args.num_sellers].train_critic(s_s, n_s_s, b_s, n_b_s)

	@tf.function
	def train_seller_actor(self, s_s, n_s_s, b_s, n_b_s, chosen_action, to_train):
		for i, seller in enumerate(self.sellers):
			if to_train is None or i == to_train:
				seller.train_actor(s_s, n_s_s, b_s, n_b_s, chosen_action)

	@tf.function
	def train_seller_critic(self, s_s, n_s_s, b_s, n_b_s, to_train):
		for i, seller in enumerate(self.sellers):
			if to_train is None or i == to_train:
				seller.train_critic(s_s, n_s_s, b_s, n_b_s)

	@tf.function
	def train_buyer_actor(self, s_s, n_s_s, b_s, n_b_s, chosen_action, to_train):
		for i, buyer in enumerate(self.buyers):
			if to_train is None or i + self.args.num_sellers == to_train:
				buyer.train_actor(s_s, n_s_s, b_s, n_b_s, chosen_action)

	@tf.function
	def train_buyer_critic(self, s_s, n_s_s, b_s, n_b_s, to_train):
		for i, buyer in enumerate(self.buyers):
			if to_train is None or i + self.args.num_sellers == to_train:
				buyer.train_critic(s_s, n_s_s, b_s, n_b_s)
	
	def train(self, args, train_id=None):
		scaled_seller_states = self.scaled_seller_states()[:self.ep_num, ..., :5].reshape((-1, args.num_sellers, 5))
		scaled_buyer_states = self.scaled_buyer_states()[:self.ep_num, ..., :11].reshape((-1, args.num_buyers, 11))
		seller_states = self.seller_states[:self.ep_num, ..., :5].reshape((-1, args.num_sellers, 5))
		buyer_states = self.buyer_states[:self.ep_num, ..., :11].reshape((-1, args.num_buyers, 11))
		
		# Buyer training
		max_id = (self.ep_num - 1) * args.steps_in_episode
		samples = np.random.randint(0, max_id, args.batch_size)
		s_s = scaled_seller_states[samples]
		n_s_s = scaled_seller_states[samples + 1]
		b_s = scaled_buyer_states[samples]
		n_b_s = scaled_buyer_states[samples + 1]
		chosen_action = buyer_states[samples, :, 4:10]
		if train_id is None:
			if self.step_num % args.actor_update_freq == 0:
				self.train_buyer_actor(s_s, n_s_s, b_s, n_b_s, chosen_action, train_id)
			self.train_buyer_critic(s_s, n_s_s, b_s, n_b_s, train_id)
		elif train_id >= args.num_sellers:
			if self.step_num % args.actor_update_freq == 0:
				self.train_single_buyer_actor(s_s, n_s_s, b_s, n_b_s, chosen_action, train_id)
			self.train_single_buyer_critic(s_s, n_s_s, b_s, n_b_s, train_id)
			
		# Seller training
		samples = np.random.randint(0, max_id, args.batch_size)
		if args.sellers_share_buffer and self.multiagent:
			permutation = np.random.permutation(args.num_sellers)
			scaled_seller_states = scaled_seller_states[:, permutation]
		s_s = scaled_seller_states[samples]
		n_s_s = scaled_seller_states[samples + 1]
		b_s = scaled_buyer_states[samples]
		n_b_s = scaled_buyer_states[samples + 1]
		chosen_action = seller_states[samples, :, 2:4]
		if train_id is None:
			if self.step_num % args.actor_update_freq == 0:
				self.train_seller_actor(s_s, n_s_s, b_s, n_b_s, chosen_action, train_id)
			self.train_seller_critic(s_s, n_s_s, b_s, n_b_s, train_id)
		elif train_id < args.num_sellers:
			if self.step_num % args.actor_update_freq == 0:
				self.train_single_seller_actor(s_s, n_s_s, b_s, n_b_s, chosen_action, train_id)
			self.train_single_seller_critic(s_s, n_s_s, b_s, n_b_s, train_id)
	
	def reward_at_end(self, args):
		good_seller = self.seller_states[self.ep_num, self.step_num, :, 1].copy()
		self.seller_states[self.ep_num, self.step_num, :, 4] += args.final_good_reward * good_seller
		
		money_buyer = self.buyer_states[self.ep_num, self.step_num, :, 2].copy()
		self.buyer_states[self.ep_num, self.step_num, :, 10] += args.final_money_reward * money_buyer
		
	def reset(self, args):
		self.ep_num += 1
		self.step_num = 0
	
	def init_single(self, batch_num):
		self.ep_num = (batch_num + 1) * self.args.eval_every
		path = os.getcwd()
		os.chdir(self.dir_name)
		self.seller_states[:self.ep_num] = np.load('seller_states.npy')[-self.ep_num:]
		self.buyer_states[:self.ep_num] = np.load('buyer_states.npy')[-self.ep_num:]
		
		os.chdir('models')
		for s in self.sellers:
			s.load(batch_num)
		
		for b in self.buyers:
			b.load(batch_num)
		
		os.chdir(path)
	
	def save(self, batch_num, models=True):
		path = os.getcwd()
		os.chdir(self.dir_name)
		
		np.save('seller_states.npy', self.seller_states[:self.ep_num])
		np.save('buyer_states.npy', self.buyer_states[:self.ep_num])
		
		if models:
			os.chdir('models')
			for b in self.buyers:
				b.save(batch_num)
		
			for s in self.sellers:
				s.save(batch_num)
		
		os.chdir(path)
	
	def save_single(self, batch, participant):
		path = os.getcwd()
		os.chdir(self.dir_name + '/models')
		
		if participant < self.args.num_sellers:
			self.sellers[participant].save(batch, eval=True)
		else:
			self.buyers[participant - self.args.num_sellers].save(batch, eval=True)
		
		os.chdir(path)
	
	def run_market(self, args, train_id=None):
		if self.ep_num >= self.seller_states.shape[0]:
			return
		
		if self.step_num % args.renew_rights_every == 0:
			self.resupply(args)
			
			self.offer_good()
			
			self.distribute_right()
		
		t = self.step_num / (args.steps_in_episode - 1)
		self.seller_states[self.ep_num, self.step_num, :, 0] = t
		self.buyer_states[self.ep_num, self.step_num, :, 0] = t
		
		self.offer_right()
		
		self.make_orders()
		
		self.trade(args)
		
		self.reward_agents(args)
		
		self.transfer_to_next_trade(args)
		
		samples = (self.ep_num - 1) * args.steps_in_episode
		if not self.eval and samples > 2 * args.batch_size:
			self.train(args, train_id)
		
		if self.step_num + 1 == args.steps_in_episode:
			self.reward_at_end(args)
			self.reset(args)
		else:
			self.step_num += 1
	
	def simulate_crisis(self, args, to_train=None):
		for _ in range(args.steps_in_episode):
			self.run_market(args, to_train)
	
	def evaluate(self, batch_num, p=None):
		self.ep_num = 0
		self.seller_states[...] = 0
		self.buyer_states[...] = 0
		
		path = os.getcwd()
		os.chdir(self.dir_name + '/models')
		for s in self.sellers:
			s.load(batch_num)
		
		for b in self.buyers:
			b.load(batch_num)
		
		if p is not None:
			if p < self.args.num_sellers:
				self.sellers[p].load(batch_num, eval=True)
			else:
				self.buyers[p - self.args.num_sellers].load(batch_num, eval=True)
			
		os.chdir(path)
		
		for _ in range(self.args.eval_for):
			for _ in range(self.args.steps_in_episode):
				self.run_market(self.args, p)

		seller_return = np.mean(np.sum(self.seller_states[..., 4], axis=1), axis=0)
		buyer_return = np.mean(np.sum(self.buyer_states[..., 10], axis=1), axis=0)
		
		return np.concatenate([seller_return, buyer_return], axis=0)
		

			
			
			