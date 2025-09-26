# agent.py

import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from constants import *

# --- AGENT CONSTANTS ---
MAX_MEMORY = 100_000
BATCH_SIZE = 512 # Increased batch size for better training stability
LR = 0.001

class Agent:
    """
    The AI agent that learns to play Snake.
    """
    def __init__(self, verbose=False):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        model_path = './model/model.keras'
        games_count_path = './model/games_count.txt'
        
        if os.path.exists(model_path):
            if verbose: print("Loading existing model and optimizer state...")
            self.model = tf.keras.models.load_model(model_path)
        else:
            if verbose: print("No existing model found. Creating a new one.")
            self.model = Linear_QNet(HIDDEN_SIZE, OUTPUT_SIZE)

        if os.path.exists(games_count_path):
            with open(games_count_path, 'r') as f:
                try:
                    self.n_games = int((f.read() or '0').strip())
                except ValueError:
                    self.n_games = 0
            if verbose: print(f"Loaded game count: {self.n_games}")

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """
        Gets the current state of the game environment.
        The state is an 11-element vector.
        """
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y  # Food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience in the memory buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        Trains the network on a random batch of experiences from memory (experience replay).
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        # Unzip the sample into separate lists
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Trains the network on the most recent experience.
        """
        self.trainer.train_step(
            np.expand_dims(state, 0),
            np.expand_dims(action, 0),
            np.expand_dims(reward, 0),
            np.expand_dims(next_state, 0),
            np.expand_dims(done, 0),
        )

    def predict_q(self, states_np):
        states = tf.convert_to_tensor(states_np, dtype=tf.float32)
        return self.model(states)

    def get_action(self, state):
        """
        Decides on an action based on the current state.
        Uses an epsilon-greedy strategy for exploration vs. exploitation.
        """
        # Random moves: tradeoff between exploration and exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = tf.convert_to_tensor(state, dtype=tf.float32)
            prediction = self.model(tf.expand_dims(state0, 0))
            move = tf.argmax(prediction[0]).numpy()
            final_move[move] = 1
        return final_move