import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.patches as mpatches
import pickle
from environment import talmud_split

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

matplotlib.use('Agg')

def my_plot(args, data, name):
	time = np.arange(data.shape[0]) + 1
	mean, std = np.mean(data, axis=-1), np.std(data, axis=-1)
	plt.plot(time, mean, 'r')
	plt.fill_between(time, mean - std, mean + std, facecolor='r', alpha=0.5, interpolate=True)
	x_label = 'Market' if data.shape[0] == args.markets_in_episode else 'Crisis'
	plt.xlabel(x_label)
	plt.ylabel(name)
	plt.ylim((0, 1.1 * np.max(mean + std)))
	plt.savefig(f'{name}.png')
	plt.clf()

def my_split_plot(args, data, name, mean=None):
	time = np.arange(data.shape[0]) + 1
	x_label = 'Market' if data.shape[0] == args.markets_in_episode else 'Crisis'
	grid_size = int(np.sqrt(args.num_buyers))
	fig, axs = plt.subplots(grid_size, grid_size)
	for i in range(grid_size):
		for j in range(grid_size):
			axs[i, j].grid()
			axs[i, j].plot(time, data[:, grid_size * i + j])
			if not mean is None:
				axs[i, j].plot(time, mean)
			
			axs[i, j].set_ylim((0, 1.1 * np.max(data)))
			if j == 0 and i == grid_size - 1:
				axs[i, j].set(xlabel=x_label, ylabel=name)
			elif j == 0:
				axs[i, j].set(ylabel=name)
			elif i == grid_size - 1:
				axs[i, j].set(xlabel=x_label)
				
	plt.savefig(f'{name}.png')
	plt.clf()



def draw(args, seller_states, buyer_states, draw_details, greedy=True):
	seller_return = np.sum(seller_states[..., 4], axis=(1, 2))
	my_plot(args, seller_return, 'seller return')
	
	buyer_return = np.sum(buyer_states[..., 10], axis=(1, 2))
	my_plot(args, buyer_return, 'buyer return')
	
	buyer_return = np.sum(buyer_states[..., 10], axis=(1, 2))
	my_split_plot(args, buyer_return, 'buyer return split')

	ending_money = buyer_states[:, -1, -1, :, 12]
	my_split_plot(args, ending_money, 'ending money split')
	
	ending_money = buyer_states[:, -1, -1, :, 12]
	my_plot(args, ending_money, 'ending money')

	ending_good = seller_states[:, -1, -1, :, 5]
	my_plot(args, ending_good, 'ending good')
	
	purchased_good = np.sum(buyer_states[..., 11], axis=(1, 2))
	my_split_plot(args, purchased_good, 'purchased good')
	
	des_price_good = np.mean(buyer_states[-1, ..., 7], axis=1)
	price_good = np.mean(seller_states[-1, ..., 3], axis=(1, 2))
	my_split_plot(args, des_price_good, 'price good', price_good)
	
	des_volume_good = np.mean(buyer_states[-1, ..., 6], axis=1)
	volume_good = np.mean(seller_states[-1, ..., 1] * seller_states[-1, ..., 2], axis=(1, 2))
	my_split_plot(args, des_volume_good, 'volume good', volume_good)
	
	des_price_good = np.mean(buyer_states[..., 7], axis=(1, 2))
	price_good = np.mean(seller_states[..., 3], axis=(1, 2, 3))
	my_split_plot(args, des_price_good, 'price good crisis', price_good)
	
	if args.fairness[-6:] == 'talmud':
		des_price_right = np.mean(buyer_states[-1, ..., 9], axis=1)
		price_right = np.mean(buyer_states[-1, ..., 5], axis=(1, 2))
		my_split_plot(args, des_price_right, 'price right', price_right)
		
		des_volume_right = np.mean(buyer_states[-1, ..., 8], axis=1)
		volume_right = np.mean(buyer_states[-1, ..., 4], axis=(1, 2))
		my_split_plot(args, des_volume_right, 'volume right', volume_right)
		
		des_price_right = np.mean(buyer_states[..., 9], axis=(1, 2))
		price_right = np.mean(buyer_states[..., 5], axis=(1, 2, 3))
		my_split_plot(args, des_price_right, 'price right crisis', price_right)


def draw_frustration(args, seller_states, buyer_states):
	frustration = np.zeros(shape=(args.eval_for, args.markets_in_episode, args.num_buyers))
	poa = np.zeros(shape=(args.markets_in_episode, args.eval_for))
	d = np.array(args.demands) / np.mean(args.demands)
	for i in range(args.eval_for):
		for market in range(args.markets_in_episode):
			offered_vol = np.sum(seller_states[i, market, 0, :, 1] * seller_states[i, market, 0, :, 2], axis=0)
			good_bought = np.zeros(shape=(args.num_buyers, ))
			for t in range(args.renew_rights_every):
				good_s = buyer_states[i, market, t, :, 1]
				good_e = buyer_states[i, market, t, :, 11]
				good_bought += good_e - good_s			
			
			good_start = buyer_states[i, market, 0, :, 1]
			need = np.maximum(d - good_start, 0)
			rights = talmud_split(need, np.sum(offered_vol))

			frustration[i, market, :] = np.clip((rights - good_bought) / rights, 0, 1)
			frustration[i, market, np.where(rights == 0)] = 0

			poa[market, i] = np.sum(frustration[i, :market+1, :]) / (args.num_buyers * (market + 1))

	my_split_plot(args, np.mean(frustration, axis=0), 'frustration')
	my_plot(args, poa, 'price of anarchy')


def produce_output(details):
	path = os.getcwd()
	args = pickle.load(open('args.pickle', 'rb'))
	seller_states = np.load('seller_states.npy').reshape((-1, args.steps_in_episode // args.renew_rights_every, args.renew_rights_every, args.num_sellers, 6))
	buyer_states = np.load('buyer_states.npy').reshape((-1, args.steps_in_episode // args.renew_rights_every, args.renew_rights_every, args.num_buyers, 14))
	nash_conv = np.load('nash_conv.npy').reshape((-1, args.num_sellers + args.num_buyers))
	
	args.markets_in_episode = args.steps_in_episode // args.renew_rights_every

	train_start = 2 * args.batch_size // (args.eval_every * args.markets_in_episode * args.renew_rights_every)
	args.eval_for = 100
	
	try:
		x = (np.arange(nash_conv.shape[0]) + 1)[train_start:] * args.eval_every
		y = np.sum(nash_conv, axis=1)[train_start:]
		mean = np.convolve(np.pad(y, (2, 2), mode='edge'), np.ones(5)/5, mode='valid')
		plt.plot(x, y, label='Nash Conv', alpha=0.5)
		plt.plot(x, mean, label='5-Running Average')
		plt.axhline(0, alpha=0.5, linestyle='--', color='gray')
		plt.legend()
		plt.xlabel('Crisis')
		plt.ylabel('Nash Conv')
		plt.savefig('nash conv.png')
		plt.clf()
	except:
		pass

	draw(args, seller_states, buyer_states, draw_details=details, greedy=False)

	draw_frustration(args, seller_states[-args.eval_for:], buyer_states[-args.eval_for:])

	os.chdir(path)
