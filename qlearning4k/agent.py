from .memory import ExperienceReplay
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from tqdm import *

class Agent:

	def __init__(self, model, memory=None, memory_size=1000, nb_frames=None):
		assert len(model.output_shape) == 2, "Model's output shape should be (nb_samples, nb_actions)."
		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)
			self.memory_inv = ExperienceReplay(memory_size) # inverted board
			self.memory_mirror = ExperienceReplay(memory_size) # mirror board
			self.memory_mirror_inv = ExperienceReplay(memory_size) # mirror inv board
		if not nb_frames and not model.input_shape[1]:
			raise Exception("Missing argument : nb_frames not provided")
		elif not nb_frames:
			nb_frames = model.input_shape[1]
		elif model.input_shape[1] and nb_frames and model.input_shape[1] != nb_frames:
			raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")
		self.model = model
		self.nb_frames = nb_frames
		self.frames = None

	@property
	def memory_size(self):
		return self.memory.memory_size

	@memory_size.setter
	def memory_size(self, value):
		self.memory.memory_size = value

	def reset_memory(self):
		self.memory.reset_memory()

	def check_game_compatibility(self, game):
		game_output_shape = (1, None) + game.get_frame().shape
		if len(game_output_shape) != len(self.model.input_shape):
			raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
		else:
			for i in range(len(self.model.input_shape)):
				if self.model.input_shape[i] and game_output_shape[i] and self.model.input_shape[i] != game_output_shape[i]:
					raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')


		if len(self.model.output_shape) != 2 or self.model.output_shape[1] != game.nb_actions:
			print('output shape:')
			print(self.model.output_shape)
			raise Exception('Output shape of model should be (nb_samples, nb_actions).')

	def get_game_data(self, game):
		frame = game.get_frame()
		if self.frames is None:
			self.frames = [frame] * self.nb_frames
		else:
			self.frames.append(frame)
			self.frames.pop(0)
		return np.expand_dims(self.frames, 0)

	def clear_frames(self):
		self.frames = None

	def train(self, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=0.5, reset_memory=False, observe=0, checkpoint=None):
		self.check_game_compatibility(game)
		if type(epsilon)  in {tuple, list}:
			delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
			final_epsilon = epsilon[1]
			epsilon = epsilon[0]
		else:
			final_epsilon = epsilon
		model = self.model
		nb_actions = model.output_shape[-1]
		win_count = 0
		for epoch in tqdm(range(nb_epoch)):
			loss = 0.
			game.reset()
			self.clear_frames()
			if reset_memory:
				self.reset_memory()
			game_over = False
			S = self.get_game_data(game)
			S_inv = game.inv_board(S)
			S_mirror_inv = game.mirror_board(S_inv)
			S_mirror = game.mirror_board(S)
			while not game_over:
				if np.random.random() < epsilon or epoch < observe:
					a = int(np.random.randint(game.nb_actions))
				else:
					q = model.predict(S)
					a = int(np.argmax(q[0]))
				game.play(a)
				oppo_action = game.oppo_action
				r = game.get_score()

				S_prime = self.get_game_data(game)
				S_prime_inv = game.inv_board(S_prime)
				S_prime_mirror_inv = game.mirror_board(S_prime_inv)
				S_prime_mirror = game.mirror_board(S_prime)

				game_over = game.is_over()

				transition = [S, a, r, S_prime, game_over]
				transition_inv = [S_inv, oppo_action, -r, S_prime_inv, game_over]
				transition_mirror_inv = [S_mirror_inv, game.mirror_action(oppo_action), -r, S_prime_mirror_inv, game_over]
				transition_mirror = [S_mirror, game.mirror_action(a), r, S_prime_mirror, game_over]

				self.memory.remember(*transition)
				self.memory_inv.remember(*transition_inv)
				self.memory_mirror_inv.remember(*transition_mirror_inv)
				self.memory_mirror.remember(*transition_mirror)

				S = S_prime
				S_mirror = S_prime_mirror
				S_inv = S_prime_inv
				S_mirror_ive = S_prime_mirror_inv

				if epoch >= observe:
					batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
					if batch:
						inputs, targets = batch
						loss += float(model.train_on_batch(inputs, targets))
					batch = self.memory_inv.get_batch(model=model, batch_size=batch_size, gamma=gamma)
					if batch:
						inputs, targets = batch
						loss += float(model.train_on_batch(inputs, targets))
					batch = self.memory_mirror.get_batch(model=model, batch_size=batch_size, gamma=gamma)
					if batch:
						inputs, targets = batch
						loss += float(model.train_on_batch(inputs, targets))
					batch = self.memory_mirror_inv.get_batch(model=model, batch_size=batch_size, gamma=gamma)
					if batch:
						inputs, targets = batch
						loss += float(model.train_on_batch(inputs, targets))
				if checkpoint and ((epoch + 1 - observe) % checkpoint == 0 or epoch + 1 == nb_epoch):
					model.save_weights('weights.hdf5')
			if game.is_won():
				win_count += 1
			if epsilon > final_epsilon and epoch >= observe:
				epsilon -= delta
			# print("Epoch {:03d}/{:03d} | Loss {:.4f} | Epsilon {:.2f} | Win count {}".format(epoch + 1, nb_epoch, loss, epsilon, win_count))

	def predict(self, game):
		model = self.model
		S = self.get_game_data(game)
		q = model.predict(S)[0]
		possible_actions = game.get_possible_actions()
		q = [q[i] for i in possible_actions]
		action = possible_actions[np.argmax(q)]
		return action

	def play(self, game, nb_epoch=10, epsilon=0., visualize=True):
		self.check_game_compatibility(game)
		model = self.model
		win_count = 0
		vis_frames = []
		for epoch in range(nb_epoch):
			game.reset()
			self.clear_frames()
			S = self.get_game_data(game)
			if visualize:
				vis_frames.append(game.draw())
			game_over = False
			while not game_over:
				if np.random.rand() < epsilon:
					print("random")
					action = int(np.random.randint(0, game.nb_actions))
				else:
					q = model.predict(S)[0]
					possible_actions = game.get_possible_actions()
					q = [q[i] for i in possible_actions]
					action = possible_actions[np.argmax(q)]
				game.play(action)
				S = self.get_game_data(game)
				if visualize:
					vis_frames.append(game.draw())
				game_over = game.is_over()
			if game.is_won():
				win_count += 1
		print("Accuracy {} %".format(100. * win_count / nb_epoch))
		if visualize:
			if 'images' not in os.listdir('.'):
				os.mkdir('images')
			for i in range(len(vis_frames)):
				plt.imshow(vis_frames[i], interpolation='none')
				# fig = plt.figure()
				# plt.plot(vis_frames[i], interpolation='none')
				plt.savefig("images/{}{:04d}.png".format(game.name,i))
