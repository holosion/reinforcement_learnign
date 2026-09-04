from collections import defaultdict
import gymnasium as gym 
import numpy as np

class BlackjackAgent:
    def __init__(
        self, 
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,):

        """Initialize a Q-learning environment
        Args:
            env: The training environment
            learning_rate: How quickly to update the Q values (0,1)
            initial_epsilon: Initial exploration rate
            epsilon_decay: How much to reduce the epsilon in each episode
            final_epsilon: Minimum exploration rate (0.1)
            discount_factor: How to value the future rewards (0-1)
        """

        self.env = env
        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor # How much we care about the future rewards


        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon


        # Track the learning process 
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy
            
        Returns
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int , bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],):

        """Update Q-values based on the reward and next state"""

        if terminated:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_values[next_obs])

        self.q_values[obs][action] = self.q_values[obs][action] + self.lr * (target - self.q_values[obs][action])