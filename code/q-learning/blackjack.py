from collections import defaultdict
import gymnasium as gym 
import numpy as np

class BlackjackAgent:
    def __init__(
        self, 
        env: gym.Env,
        learning_rate: float,
        inital_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,):

        """initialize a q-learning environment
        args:
        env: the trainig environment
        learning_rate: how quickly to update the q values (0,1)
        epsilon_decay: how much to reduce the epsilon in each episode
        final_epsilon: minimum exploration rate(0.1)
        discount_factor: how to value the furtrure rewards (0-1)
        """

        self.env = env
        #q-table : maps  (state, action) to expected reward
        # defaultdict automatically creaes entriews with zeros for new states

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor # how much we care about the future rewards


        #exploration parameters
        self.epsilon = inital_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_position = final_epsilon


        #track the learning process 
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """choose an action using epsilon-greedy strategy
        
        Returns
            aciton: 0(stand) or 1 (hit)
            
            """
        #with probablility epsilon:explore (random aciton)
        if np.random.random() < self.epsilon:
            return self.env.acton_space.sample()

        #with probability (1-epsilon):exploit(best for known action)
        else:
            return int(np.argmax(self.q_value[obs]))

    def update(
        self,
        obs: tuple[int, int , bool],
        action: int,
        reward: float,
        terminated: bool,
        next-obs: tuple[int, int, bool],):

        """"""



        