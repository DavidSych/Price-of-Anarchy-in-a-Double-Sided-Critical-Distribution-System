import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import pickle
from environment import talmud_split as cg_split
import os
import argparse

mpl.use('Agg')


def my_plot(args, data, name, fair):
	time = np.arange(data.shape[0]) + 2  # number of agents actually
	mean, std = np.mean(data, axis=-1), np.std(data, axis=-1)
	if fair == 'free':
		plt.fill_between(time, mean - std, mean + std, facecolor='royalblue', alpha=0.5, interpolate=True)
		plt.plot(time, mean, 'royalblue', linestyle='-', label='Free market, k=1')
	elif fair == 'talmud':
		plt.fill_between(time, mean - std, mean + std, facecolor='chocolate', alpha=0.5, interpolate=True)
		plt.plot(time, mean, 'chocolate', linestyle='--', label='Rights, k=1')
	elif fair == '2talmud':
		plt.fill_between(time, mean - std, mean + std, facecolor='forestgreen', alpha=0.5, interpolate=True)
		plt.plot(time, mean, 'forestgreen', linestyle='-.', label='Rights, k=2')
	else:
		raise NotImplementedError(f'Unknown fairness {fair}')
	x_label = 'Number of Buyers and Sellers'
	plt.xlabel(x_label, fontsize=fontsize+4)
	plt.ylabel(name, fontsize=fontsize+4)
	plt.ylim(0, 1)


def get_final_poa(args, seller_states, buyer_states):
	frustration = np.zeros(shape=(args.eval_for, args.markets_in_episode, args.num_buyers))
	poa = np.zeros(shape=(args.markets_in_episode, args.eval_for))
	d = np.array(args.demands) / np.mean(args.demands)
	for i in range(args.eval_for):
		for market in range(args.markets_in_episode):
			offered_vol = np.sum(seller_states[i, market, 0, :, 1] * seller_states[i, market, 0, :, 2], axis=0)
			good_bought = np.zeros(shape=(args.num_buyers,))
			for t in range(args.renew_rights_every):
				good_s = buyer_states[i, market, t, :, 1]
				good_e = buyer_states[i, market, t, :, 11]
				good_bought += good_e - good_s

			good_start = buyer_states[i, market, 0, :, 1]
			need = np.maximum(d - good_start, 0)
			rights = cg_split(need, np.sum(offered_vol))

			frustration[i, market, :] = np.clip((rights - good_bought) / rights, 0, 1)
			frustration[i, market, np.where(rights == 0)] = 0

			poa[market, i] = np.sum(frustration[i, :market + 1, :]) / (args.num_buyers * (market + 1))

	return np.mean(poa, axis=1)[-1]

def produce_poa(mech):
	root = os.getcwd()
	plt.grid(alpha=0.25)
	for f in ['free', 'talmud', '2talmud']:
		poa = np.zeros((18, 10))
		for n in range(2, 20):
			for directory in os.listdir():
				print(directory)
				if not os.path.isdir(directory):
					continue
				_, mechanism, fairness, seed, num_agents = directory.split('_')
				if len(num_agents) == 0:
					continue
				if int(num_agents) == n and mechanism == mech and fairness == f:
					os.chdir(directory)
					args = pickle.load(open('args.pickle', 'rb'))
					seller_states = np.load('seller_states.npy').reshape((-1, args.steps_in_episode // args.renew_rights_every, args.renew_rights_every, args.num_sellers, 6))
					buyer_states = np.load('buyer_states.npy').reshape((-1, args.steps_in_episode // args.renew_rights_every, args.renew_rights_every, args.num_buyers, 14))
	
					args.markets_in_episode = args.steps_in_episode // args.renew_rights_every
					args.eval_for = 100
	
					poa[n - 2, int(seed) - 1] = get_final_poa(args, seller_states[-args.eval_for:], buyer_states[-args.eval_for:])
	
					os.chdir(root)

		my_plot(args, poa, 'Price of Anarchy', f)


parser = argparse.ArgumentParser()
parser.add_argument("--mechanism", default='none', type=str, help="Mechanism to draw.")
args = parser.parse_args([] if "__file__" not in globals() else None)

fontsize = 16
mech = args.mechanism
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gcf().subplots_adjust(bottom=0.2, left=0.15)
produce_poa(mech=mech)
plt.legend(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

if mech == 'random':
	title = 'Random'
elif mech == 'greedy':
	title = 'Greedy'
elif mech == 'average':
	title = 'Maximum clearing (average)'
elif mech == 'absolute':
	title = 'Maximum clearing (absolute)'

plt.title(title, fontsize=fontsize+6)
plt.savefig(f'output_{mech}_poa_num_agents.pdf')
plt.clf()





