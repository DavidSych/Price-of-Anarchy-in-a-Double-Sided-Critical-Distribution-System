import tensorflow as tf
import tensorflow_probability as tfp


class Seller():
	def __init__(self, args, i):
		self.args, self.id = args, i

		self.get_actor(args)
		self.get_critic(args)
	
	def get_actor(self, args):
		my_state = tf.keras.Input(shape=(1,))
		buyer_states = tf.keras.Input(shape=(args.num_buyers, 2))

		flat_buyer_states = tf.keras.layers.Flatten()(buyer_states)
		hidden = tf.keras.layers.Concatenate()([my_state, flat_buyer_states])
		hidden = tf.keras.layers.Dense(args.seller_hidden_actor, activation='relu')(hidden)
		hidden = tf.keras.layers.Dense(args.seller_hidden_actor, activation='relu')(hidden)
		
		mu = tf.keras.layers.Dense(2, activation='sigmoid')(hidden)
		sigma = tf.keras.layers.Dense(2, activation='softplus')(hidden) + 1e-4
		
		self.actor = tf.keras.Model(inputs={'my state': my_state, 'buyer states': buyer_states},
									outputs={'m': mu, 's': sigma})
		
		self.actor.compile(optimizer=tf.keras.optimizers.Adam(args.actor_learning_rate, global_clipnorm=args.clip_norm))
		self.target_actor = tf.keras.models.clone_model(self.actor)
		
	def get_critic(self, args):
		my_state = tf.keras.Input(shape=(2,))
		my_action = tf.keras.Input(shape=(2,))
		seller_states = tf.keras.Input(shape=(args.num_sellers - 1, 3))
		buyer_states = tf.keras.Input(shape=(args.num_buyers, 9))
		
		flat_seller_states = tf.keras.layers.Flatten()(seller_states)
		flat_buyer_states = tf.keras.layers.Flatten()(buyer_states)
		hidden = tf.keras.layers.Concatenate()([my_state, flat_seller_states, flat_buyer_states])
		
		x = tf.keras.layers.Dense(args.seller_hidden_critic, activation='relu')(hidden)
		x = tf.keras.layers.Dense(args.seller_hidden_critic, activation='relu')(x)

		y = tf.keras.layers.Dense(args.seller_hidden_critic, activation='relu')(hidden)
		y = tf.keras.layers.Dense(args.seller_hidden_critic, activation='relu')(y)
		
		value = tf.minimum(x, y)[:, 0]
		
		self.critic = tf.keras.Model(inputs={'my state': my_state, 'my action': my_action,
											 'seller states': seller_states, 'buyer states': buyer_states},
									 outputs=value)
		
		self.critic.compile(optimizer=tf.keras.optimizers.Adam(args.critic_learning_rate, global_clipnorm=args.clip_norm))
		self.target_critic = tf.keras.models.clone_model(self.critic)

	def train_critic(self, seller_state, next_seller_state, buyer_state, next_buyer_state):
		dones = tf.cast(seller_state[:, self.id, 0] >= 1, tf.float32)
		if self.args.reward_clipping:
			rewards = tf.clip_by_value(seller_state[:, self.id, 4], - self.args.clip_const, self.args.clip_const)
		else:
			rewards = seller_state[:, self.id, 4]
		returns = rewards + self.args.gamma * (1 - dones) * self.value(next_seller_state, next_buyer_state)
		
		my_state = seller_state[:, self.id, :2]
		my_action = seller_state[:, self.id, 2:4]
		other_seller_states = tf.concat([seller_state[..., :self.id, 1:4], seller_state[..., self.id + 1:, 1:4]], axis=-2)
		buyer_states = buyer_state[..., 1:10]

		with tf.GradientTape() as tape:
			value = self.critic({'my state': my_state, 'my action': my_action,
								 'seller states': other_seller_states, 'buyer states': buyer_states})
			
			loss = tf.keras.losses.MeanSquaredError()(y_true=returns, y_pred=value)
			
			for var in self.critic.trainable_variables:
				loss += self.args.l2 * tf.nn.l2_loss(var)

		self.critic.optimizer.minimize(loss, self.critic.trainable_variables, tape=tape)

	def train_actor(self, seller_state, next_seller_state, buyer_state, next_buyer_state, chosen_action):
		dones = tf.cast(seller_state[:, self.id, 0] >= 1, tf.float32)
		if self.args.reward_clipping:
			rewards = tf.clip_by_value(seller_state[:, self.id, 4], - self.args.clip_const, self.args.clip_const)
		else:
			rewards = seller_state[:, self.id, 4]
		returns = rewards + self.args.gamma * (1 - dones) * self.value(next_seller_state, next_buyer_state)
		values = self.value(seller_state, buyer_state)
		
		if self.args.upgoing_policy:
			advantage = tf.maximum(returns - values, 0)
		else:
			advantage = returns - values
		
		with tf.GradientTape() as tape:
			pred = self.actor({'my state': seller_state[:, self.id, 1:2], 'buyer states': buyer_state[..., 1:3]})
			dist = tfp.distributions.Normal(pred['m'], pred['s'])
			
			loss = tf.reduce_mean(tf.math.reduce_sum(-dist.log_prob(chosen_action[:, self.id, :]), axis=1) * advantage)
			
			loss += -self.args.entropy_regularization_seller * tf.reduce_mean(tf.reduce_sum(dist.entropy(), axis=1))
			
			for var in self.critic.trainable_variables:
				loss += self.args.l2 * tf.nn.l2_loss(var)

		self.actor.optimizer.minimize(loss, self.actor.trainable_variables, tape=tape)
		
		for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
			target_var.assign(target_var * (1 - self.args.target_tau) + var * self.args.target_tau)

		for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
			target_var.assign(target_var * (1 - self.args.target_tau) + var * self.args.target_tau)

	def action(self, my_state, buyer_states):
		preds = self.target_actor({'my state': my_state, 'buyer states': buyer_states})
		dist = tfp.distributions.Normal(preds['m'], preds['s'])
		return tf.clip_by_value(dist.sample(), 0, 1)
	
	def value(self, seller_state, buyer_state):
		my_state = seller_state[:, self.id, 1:2]
		buyer_stock = buyer_state[..., 1:3]
		action = self.target_actor({'my state': my_state, 'buyer states': buyer_stock})
		
		full_state = seller_state[:, self.id, :2]
		other_seller_states = tf.concat([seller_state[..., :self.id, 1:4], seller_state[..., self.id+1:, 1:4]], axis=-2)
		buyer_states = buyer_state[..., 1:10]
		
		if self.args.train_noise_clip > 0:
			dist = tfp.distributions.Normal(action['m'], self.args.train_noise_var)
			sampled_action = tf.clip_by_value(dist.sample(), -self.args.train_noise_clip, self.args.train_noise_clip)
		else:
			sampled_action = action['m']
		
		vol_good = sampled_action[:, 0] * seller_state[:, self.id, 1]
		prc_good = sampled_action[:, 1] * self.args.max_trade_price
		scaled_sampled_action = tf.concat([vol_good[:, tf.newaxis], prc_good[:, tf.newaxis]], axis=1)
		
		value = self.target_critic({'my state': full_state, 'my action': scaled_sampled_action,
									'seller states': other_seller_states, 'buyer states': buyer_states})
		
		return value
	
	def save(self, ep_num, eval=False):
		if eval:
			self.actor.save_weights(f's_actor_{self.id}_{ep_num}_1.h5')
			self.target_actor.save_weights(f's_target_actor_{self.id}_{ep_num}_1.h5')
			self.critic.save_weights(f's_critic_{self.id}_{ep_num}_1.h5')
			self.target_critic.save_weights(f's_target_critic_{self.id}_{ep_num}_1.h5')
		else:
			self.actor.save_weights(f's_actor_{self.id}_{ep_num}_0.h5')
			self.target_actor.save_weights(f's_target_actor_{self.id}_{ep_num}_0.h5')
			self.critic.save_weights(f's_critic_{self.id}_{ep_num}_0.h5')
			self.target_critic.save_weights(f's_target_critic_{self.id}_{ep_num}_0.h5')
	
	def load(self, ep_num, eval=False):
		if eval:
			self.actor.load_weights(f's_actor_{self.id}_{ep_num}_1.h5')
			self.target_actor.load_weights(f's_target_actor_{self.id}_{ep_num}_1.h5')
			self.critic.load_weights(f's_critic_{self.id}_{ep_num}_1.h5')
			self.target_critic.load_weights(f's_target_critic_{self.id}_{ep_num}_1.h5')
		else:
			self.actor.load_weights(f's_actor_{self.id}_{ep_num}_0.h5')
			self.target_actor.load_weights(f's_target_actor_{self.id}_{ep_num}_0.h5')
			self.critic.load_weights(f's_critic_{self.id}_{ep_num}_0.h5')
			self.target_critic.load_weights(f's_target_critic_{self.id}_{ep_num}_0.h5')


class Buyer():
	def __init__(self, args, i):
		self.args, self.id = args, i

		self.get_actor(args)
		self.get_critic(args)
		
	def get_actor(self, args):
		my_state = tf.keras.Input(shape=(3,))
		seller_states = tf.keras.Input(shape=(args.num_sellers, 2))
		buyer_states = tf.keras.Input(shape=(args.num_buyers - 1, 2))
		
		flat_seller_states = tf.keras.layers.Flatten()(seller_states)
		input1 = tf.keras.layers.Concatenate()([my_state, flat_seller_states])
		
		x = tf.keras.layers.Dense(args.buyer_hidden_actor, activation='relu')(input1)
		
		mu_sell = tf.keras.layers.Dense(2, activation='sigmoid')(x)
		sigma_sell = tf.keras.layers.Dense(2, activation='softplus')(x) + 1e-4

		input2 = tf.keras.layers.Flatten()(buyer_states)
		y = tf.keras.layers.Concatenate()([input1, input2])
		
		y = tf.keras.layers.Dense(args.buyer_hidden_actor, activation='relu')(y)
		
		mu_buy = tf.keras.layers.Dense(4, activation='sigmoid')(y)
		sigma_buy = tf.keras.layers.Dense(4, activation='softplus')(y) + 1e-4
		
		self.actor = tf.keras.Model(inputs={'my state': my_state, 'seller states': seller_states, 'buyer states': buyer_states},
									outputs={'m sell': mu_sell, 's sell': sigma_sell, 'm buy': mu_buy, 's buy': sigma_buy})
		
		self.actor.compile(optimizer=tf.keras.optimizers.Adam(args.actor_learning_rate, global_clipnorm=args.clip_norm))
		self.target_actor = tf.keras.models.clone_model(self.actor)
	
	def get_critic(self, args):
		my_state = tf.keras.Input(shape=(10,))
		seller_states = tf.keras.Input(shape=(args.num_sellers, 3))
		buyer_states = tf.keras.Input(shape=(args.num_buyers - 1, 9))
		
		flat_seller_states = tf.keras.layers.Flatten()(seller_states)
		flat_buyer_states = tf.keras.layers.Flatten()(buyer_states)
		hidden = tf.keras.layers.Concatenate()([my_state, flat_seller_states, flat_buyer_states])
		
		x = tf.keras.layers.Dense(args.buyer_hidden_critic, activation='relu')(hidden)
		x = tf.keras.layers.Dense(args.buyer_hidden_critic, activation='relu')(x)
		
		y = tf.keras.layers.Dense(args.buyer_hidden_critic, activation='relu')(hidden)
		y = tf.keras.layers.Dense(args.buyer_hidden_critic, activation='relu')(y)
		
		value = tf.minimum(x, y)[:, 0]
		
		self.critic = tf.keras.Model(inputs={'my state': my_state, 'seller states': seller_states, 'buyer states': buyer_states},
									 outputs=value)
		
		self.critic.compile(optimizer=tf.keras.optimizers.Adam(args.critic_learning_rate, global_clipnorm=args.clip_norm))
		self.target_critic = tf.keras.models.clone_model(self.critic)

	def train_critic(self, seller_state, next_seller_state, buyer_state, next_buyer_state):
		dones = tf.cast(buyer_state[:, self.id, 0] >= 1, tf.float32)
		if self.args.reward_clipping:
			rewards = tf.clip_by_value(buyer_state[:, self.id, 10], - self.args.clip_const, self.args.clip_const)
		else:
			rewards = buyer_state[:, self.id, 10]
		returns = rewards + self.args.gamma * (1 - dones) * self.value(next_seller_state, next_buyer_state)
		
		my_full_state = buyer_state[:, self.id, :10]
		seller_states = seller_state[..., 1:4]
		other_buyer_states = tf.concat([buyer_state[..., :self.id, 1:10], buyer_state[..., self.id + 1:, 1:10]], axis=-2)

		with tf.GradientTape() as tape:
			value = self.critic({'my state': my_full_state, 'seller states': seller_states,
								 'buyer states': other_buyer_states})
			
			loss = tf.keras.losses.MeanSquaredError()(y_pred=value, y_true=returns)
			
			for var in self.critic.trainable_variables:
				loss += self.args.l2 * tf.nn.l2_loss(var)

		self.critic.optimizer.minimize(loss, self.critic.trainable_variables, tape=tape)

	def train_actor(self, seller_state, next_seller_state, buyer_state, next_buyer_state, chosen_action):
		dones = tf.cast(buyer_state[:, self.id, 0] >= 1, tf.float32)
		if self.args.reward_clipping:
			rewards = tf.clip_by_value(buyer_state[:, self.id, 10], - self.args.clip_const, self.args.clip_const)
		else:
			rewards = buyer_state[:, self.id, 10]
		returns = rewards + self.args.gamma * (1 - dones) * self.value(next_seller_state, next_buyer_state)
		values = self.value(seller_state, buyer_state)
		
		if self.args.upgoing_policy:
			advantage = tf.maximum(returns - values, 0)
		else:
			advantage = returns - values
			
		actions_sell = chosen_action[:, self.id, :2]
		actions_buy = chosen_action[:, self.id, 2:]
		
		seller_offers = seller_state[:, :, 2:4]
		my_state = buyer_state[:, self.id, 1:4]
		right_offers = tf.concat([buyer_state[:, :self.id, 4:6], buyer_state[:, self.id+1:, 4:6]], axis=1)
		
		with tf.GradientTape() as tape:
			pred = self.actor({'my state': my_state, 'buyer states': right_offers, 'seller states': seller_offers})
			dist_sell = tfp.distributions.Normal(pred['m sell'], pred['s sell'])
			dist_buy = tfp.distributions.Normal(pred['m buy'], pred['s buy'])
			loss = tf.reduce_mean(tf.math.reduce_sum(-dist_sell.log_prob(actions_sell), axis=1) * advantage)
			loss += tf.reduce_mean(tf.math.reduce_sum(-dist_buy.log_prob(actions_buy), axis=1) * advantage)

			loss += -self.args.entropy_regularization_seller * tf.reduce_mean(tf.reduce_sum(dist_sell.entropy(), axis=1))
			loss += -self.args.entropy_regularization_seller * tf.reduce_mean(tf.reduce_sum(dist_buy.entropy(), axis=1))

			for var in self.critic.trainable_variables:
				loss += self.args.l2 * tf.nn.l2_loss(var)

		self.actor.optimizer.minimize(loss, self.actor.trainable_variables, tape=tape)
		
		for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
			target_var.assign(target_var * (1 - self.args.target_tau) + var * self.args.target_tau)

		for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
			target_var.assign(target_var * (1 - self.args.target_tau) + var * self.args.target_tau)

	def sell_action(self, seller_states, buyer_states):
		my_state = buyer_states[:, self.id]
		zeros = tf.zeros(shape=(1, self.args.num_buyers - 1, 2))
		preds = self.target_actor({'my state': my_state, 'seller states': seller_states, 'buyer states': zeros})
		dist = tfp.distributions.Normal(preds['m sell'], preds['s sell'])
		return tf.clip_by_value(dist.sample(), 0, 1)

	def buy_action(self, seller_states, buyer_states):
		my_state = buyer_states[:, self.id, :3]
		rights = tf.concat([buyer_states[..., :self.id, 3:5], buyer_states[..., self.id+1:, 3:5]], axis=-2)
		preds = self.target_actor({'my state': my_state, 'seller states': seller_states, 'buyer states': rights})
		dist = tfp.distributions.Normal(preds['m buy'], preds['s buy'])
		return tf.clip_by_value(dist.sample(), 0, 1)
	
	def value(self, seller_state, buyer_state):
		my_state = buyer_state[:, self.id, 1:4]
		seller_offers = seller_state[:, :, 2:4]
		right_offers = tf.concat([buyer_state[..., :self.id, 4:6], buyer_state[..., self.id+1:, 4:6]], axis=-2)
		action = self.target_actor({'my state': my_state, 'seller states': seller_offers, 'buyer states': right_offers})
		
		seller_states = seller_state[..., 1:4]
		other_buyer_states = tf.concat([buyer_state[..., :self.id, 1:10], buyer_state[..., self.id+1:, 1:10]], axis=-2)
		
		if self.args.train_noise_clip > 0:
			dist_sell = tfp.distributions.Normal(action['m sell'], self.args.train_noise_var)
			sampled_action_sell = tf.clip_by_value(dist_sell.sample(), -self.args.train_noise_clip, self.args.train_noise_clip)
			dist_buy = tfp.distributions.Normal(action['m buy'], self.args.train_noise_var)
			sampled_action_buy = tf.clip_by_value(dist_buy.sample(), -self.args.train_noise_clip, self.args.train_noise_clip)
		else:
			sampled_action_sell = action['m sell']
			sampled_action_buy = action['m buy']
		
		vol_offer = sampled_action_sell[:, 0] * buyer_state[:, self.id, 3]
		prc_offer = sampled_action_sell[:, 1] * self.args.max_trade_price
		vol_desire_g = sampled_action_buy[:, 0] * self.args.max_trade_volume
		prc_desire_g = sampled_action_buy[:, 1] * self.args.max_trade_price
		vol_desire_r = sampled_action_buy[:, 2] * self.args.max_trade_volume
		prc_desire_r = sampled_action_buy[:, 3] * self.args.max_trade_price
		scaled_sampled_action = tf.concat([vol_offer[:, tf.newaxis], prc_offer[:, tf.newaxis],
										   vol_desire_g[:, tf.newaxis], prc_desire_g[:, tf.newaxis],
										   vol_desire_r[:, tf.newaxis], prc_desire_r[:, tf.newaxis]], axis=1)

		my_full_state = tf.concat([buyer_state[:, self.id, :4], scaled_sampled_action], axis=1)
		value = self.target_critic({'my state': my_full_state, 'seller states': seller_states,
									'buyer states': other_buyer_states})
		
		return value
	
	def save(self, ep_num, eval=False):
		if eval:
			self.actor.save_weights(f'b_actor_{self.id}_{ep_num}_1.h5')
			self.target_actor.save_weights(f'b_target_actor_{self.id}_{ep_num}_1.h5')
			self.critic.save_weights(f'b_critic_{self.id}_{ep_num}_1.h5')
			self.target_critic.save_weights(f'b_target_critic_{self.id}_{ep_num}_1.h5')
		else:
			self.actor.save_weights(f'b_actor_{self.id}_{ep_num}_0.h5')
			self.target_actor.save_weights(f'b_target_actor_{self.id}_{ep_num}_0.h5')
			self.critic.save_weights(f'b_critic_{self.id}_{ep_num}_0.h5')
			self.target_critic.save_weights(f'b_target_critic_{self.id}_{ep_num}_0.h5')
	
	def load(self, ep_num, eval=False):
		if eval:
			self.actor.load_weights(f'b_actor_{self.id}_{ep_num}_1.h5')
			self.target_actor.load_weights(f'b_target_actor_{self.id}_{ep_num}_1.h5')
			self.critic.load_weights(f'b_critic_{self.id}_{ep_num}_1.h5')
			self.target_critic.load_weights(f'b_target_critic_{self.id}_{ep_num}_1.h5')
		else:
			self.actor.load_weights(f'b_actor_{self.id}_{ep_num}_0.h5')
			self.target_actor.load_weights(f'b_target_actor_{self.id}_{ep_num}_0.h5')
			self.critic.load_weights(f'b_critic_{self.id}_{ep_num}_0.h5')
			self.target_critic.load_weights(f'b_target_critic_{self.id}_{ep_num}_0.h5')