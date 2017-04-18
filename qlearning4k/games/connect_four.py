# connect four

import numpy as np
import random

class Connect(object):

	def __init__(self, m, n):
		self.m = m
		self.n = n
		self.reset()

	@property
	def name(self):
		return "Connect Four"

	@property
	def nb_actions(self):
		return self.n

	def reset(self):
		self.noX = random.randint(0, self.m-1)
		self.noY = random.randint(0, self.n-1)
		self.gs = 'playing'
		self.board = np.zeros((self.m, self.n))
		self.top = np.full((self.n), self.m-1)
		self.board[self.noX][self.noY] = 3
		if self.noX == self.m-1:
			self.top[self.noY] -= 1

	def play(self, action):
		if self.top[action] < 0:
			self.gs = 'illegal step'
			return

		x = self.top[action]
		y = action
		self.board[x][y] = 2

		if self.has_one_side_win(2, x, y):
			self.gs = 'machine win'
			return

		# update top
		self.top[y] -= 1
		if self.noX == self.top[y] and self.noY == y:
			self.top[y] -= 1

		# pseudo-user play
		available_actions = []
		for idx, c in enumerate(self.top):
			if c >= 0:
				available_actions.append(idx)
		if len(available_actions) == 0:
			self.gs = 'tie'
			return
		# random play
		y = random.choice(available_actions)
		x = self.top[y]
		self.board[x][y] = 1

		if self.has_one_side_win(1, x, y):
			self.gs = 'user win'
			return

		# update top
		self.top[y] -= 1
		if self.noX == self.top[y] and self.noY == y:
			self.top[y] -= 1
		available_actions = []
		for idx, c in enumerate(self.top):
			if c >= 0:
				available_actions.append(idx)
		if len(available_actions) == 0:
			self.gs = 'tie'
			return

	def get_state(self):
		return self.board

	def get_score(self):
		score_map = {
			'playing': 0,
			'tie': 0,
			'machine win': 1,
			'user win': -1,
			'illegal step': -1
		}
		return score_map[self.gs]

	def is_over(self):
		return self.gs != 'playing'

	def has_one_side_win(self, pawn, x, y):
		# horizontal
		dots = 0
		for i in range(y, y+4):
			if i >= self.n or self.board[x][i] != pawn:
				break
			dots += 1
		for i in reversed(range(y-3, y)):
			if i < 0 or self.board[x][i] != pawn:
				break
			dots += 1
		if dots >= 4:
			return True

		# vertical
		dots = 0
		for i in range(x, x+4):
			if i >= self.m or self.board[i][y] != pawn:
				break
			dots += 1
		for i in reversed(range(x-3, x)):
			if i < 0 or self.board[i][y] != pawn:
				break
			dots += 1
		if dots >= 4:
			return True

		# lt->rb
		dots = 0
		for i in range(0, 4):
			if x+i >= self.m or y+i >= self.n or self.board[x+i][y+i] != pawn:
				break
			dots += 1
		for i in reversed(range(1,4)):
			if x-i < 0 or y-i < 0 or self.board[x-i][y-i] != pawn:
				break
			dots += 1
		if dots >= 4:
			return True

		# rt->lb
		dots = 0
		for i in range(0, 4):
			if x-i < 0 or y+i >= self.n or self.board[x-i][y+i] != pawn:
				break
			dots += 1
		for i in reversed(range(1,4)):
			if x+i >= self.m or y-i < 0 or self.board[x+i][y-i] != pawn:
				break
			dots += 1
		if dots >= 4:
			return True

		return False

	def is_won(self):
		return self.gs == 'machine win'

	def get_frame(self):
		return self.get_state()

	def draw(self):
		return np.array(self.get_state())

	def get_possible_actions(self):
		return range(self.nb_actions)
