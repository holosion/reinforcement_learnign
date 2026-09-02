"""Play the game of Nim against a trained computer player."""

import random


class Nim:
	def __init__(self, initial=None):
		self.piles = initial[:] if initial is not None else [1, 3, 5, 7]
		self.player = 0

	def available_actions(self):
		return {
			(pile, count)
			for pile, pile_size in enumerate(self.piles)
			for count in range(1, pile_size + 1)
		}

	def move(self, pile, count):
		if (pile, count) not in self.available_actions():
			raise ValueError("That move is not available.")
		self.piles[pile] -= count
		self.player = 1 - self.player

	def is_over(self):
		return all(pile == 0 for pile in self.piles)


class NimAI:
	def __init__(self, alpha=0.5, epsilon=0.1):
		self.q = {}
		self.alpha = alpha
		self.epsilon = epsilon

	def get_q_value(self, state, action):
		return self.q.get((tuple(state), action), 0.0)

	def best_future_reward(self, state, actions):
		if not actions:
			return 0.0
		return max(self.get_q_value(state, action) for action in actions)

	def update(self, old_state, action, new_state, reward, new_actions):
		old_q = self.get_q_value(old_state, action)
		future_reward = self.best_future_reward(new_state, new_actions)
		self.q[(tuple(old_state), action)] = old_q + self.alpha * (
			reward + future_reward - old_q
		)

	def choose(self, state, actions, epsilon=True):
		actions = list(actions)
		if not actions:
			return None
		if epsilon and random.random() < self.epsilon:
			return random.choice(actions)
		best_value = max(self.get_q_value(state, action) for action in actions)
		best_actions = [
			action
			for action in actions
			if self.get_q_value(state, action) == best_value
		]
		return random.choice(best_actions)


def train(n_episodes=10_000):
	"""Train an AI by playing Nim against itself."""
	ai = NimAI()

	for _ in range(n_episodes):
		game = Nim()
		last_state = {0: None, 1: None}
		last_action = {0: None, 1: None}
		last_player = None

		while not game.is_over():
			state = game.piles[:]
			player = game.player
			action = ai.choose(state, game.available_actions())
			game.move(*action)

			if last_player is not None:
				ai.update(
					last_state[last_player],
					last_action[last_player],
					state,
					-1,
					game.available_actions(),
				)

			last_state[player] = state
			last_action[player] = action
			last_player = player

		ai.update(last_state[last_player], last_action[last_player], [], 1, [])

	return ai


def play(ai):
	"""Run a human-versus-AI console game."""
	game = Nim()
	human_player = random.randint(0, 1)
	print("Welcome to Nim. Remove one or more objects from one pile.")
	print(f"You are player {human_player}. The computer goes first if you are player 1.")

	while not game.is_over():
		print(f"\nPiles: {game.piles}")
		if game.player == human_player:
			while True:
				try:
					pile = int(input("Choose a pile (1-4): ")) - 1
					count = int(input("How many objects: "))
					game.move(pile, count)
					break
				except (ValueError, IndexError):
					print("Invalid move. Choose a non-empty pile and a valid count.")
		else:
			action = ai.choose(game.piles, game.available_actions(), epsilon=False)
			print(f"Computer removes {action[1]} from pile {action[0] + 1}.")
			game.move(*action)

	winner = 1 - game.player
	print(f"\nPlayer {winner} made the last move.")
	print("You win!" if winner == human_player else "Computer wins!")


if __name__ == "__main__":
	play(train())