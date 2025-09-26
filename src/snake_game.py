# snake_game.py

import numpy as np
import pygame
import random
from enum import Enum
from collections import namedtuple
from constants import *

# Initialize Pygame
pygame.init()
font = pygame.font.Font(None, 25)

# Define an enumeration for directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define a simple Point data structure
Point = namedtuple('Point', 'x, y')
def manhattan(a, b):  # a, b are Points
        return abs(a.x - b.x) + abs(a.y - b.y)

BASE_TURN_PENALTY = -1
TAIL_PROXIMITY_PENALTY = -0.5 # multiplies by num of tail segments getting closer
FOOD_REWARD = 30
FOOD_PROXIMITY_REWARD = 1
FOOD_PROXIMITY_PENALTY = -2
DEATH_PENALTY = -1000

class SnakeGameAI:
    """
    The main class for the Snake game environment.
    """
    def __init__(self, w=640, h=480, render=True, render_every=1):
        self.w, self.h, self.render = w, h, render
        self.clock = pygame.time.Clock()
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
        else:
            # Offscreen surface so draw calls are safe if accidentally called
            self.display = pygame.Surface((self.w, self.h))
        self.render_every = render_every
        self._frame_render_counter = 0
        self.viewer_best = 0
        self.last_best = 0         # the best *before* the most recent improvement
        self.reset()
    
    def reset(self):
        """
        Resets the game to its initial state.
        """
        # Init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.distance_to_food = manhattan(self.head, self.food)
        self.consecutive_turns = 0
    
    def _place_food(self):
        """
        Places food at a random location on the screen, not inside the snake.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """
        Takes an action from the AI and advances the game by one frame.
        :param action: A list representing the action [straight, right, left]
        :return: A tuple (reward, game_over, score)
        """
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        reward = 0

        if not np.array_equal(action, [1, 0, 0]): # If it's a turn
            self.consecutive_turns += 1
            if(self.consecutive_turns == 2 and self.score > 15):
                reward = abs(BASE_TURN_PENALTY) * self.consecutive_turns * 1.1
            else:
                reward = BASE_TURN_PENALTY * self.consecutive_turns
        else: # If it moves straight
            self.consecutive_turns = 0

        old_head = self.head
        tail_to_consider = self.snake[3:]

        self._move(action)
        self.snake.insert(0, self.head)

        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = DEATH_PENALTY

            if self.score > self.viewer_best:
                self.last_best = self.viewer_best
                self.viewer_best = self.score
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = FOOD_REWARD * (15/self.score)
            self._place_food()
            self.distance_to_food = manhattan(self.head, self.food)
            if self.score > self.viewer_best:
                self.viewer_best = self.score
        else:
            self.snake.pop()
            
            tail_penalty = 0
            if len(tail_to_consider) > 0:
                closer = 0
                ohx, ohy = int(old_head.x), int(old_head.y)
                hx, hy = int(self.head.x), int(self.head.y)
                for seg in tail_to_consider:
                    sx, sy = int(seg.x), int(seg.y)
                    if (abs(hx - sx) + abs(hy - sy)) < (abs(ohx - sx) + abs(ohy - sy)):
                        closer += 1
                tail_penalty = closer * TAIL_PROXIMITY_PENALTY
            reward += tail_penalty

            new_distance_to_food = manhattan(self.head, self.food)
            if new_distance_to_food < self.distance_to_food:
                reward += FOOD_PROXIMITY_REWARD
            else:
                reward += FOOD_PROXIMITY_PENALTY
            self.distance_to_food = new_distance_to_food
        
        self._update_ui()
        if self.render:
            self.clock.tick(SPEED)
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        """
        Checks for collisions with walls or the snake's own body.
        :param pt: An optional point to check for collision (defaults to snake's head)
        :return: True if a collision occurs, False otherwise
        """
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        if not self.render:
            return
        self._frame_render_counter = (self._frame_render_counter + 1) % self.render_every
        if self._frame_render_counter != 0:
            return
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # (Optional) keep current score on the left
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, [0, 0])

        # --- Top-right: Best + Last Best (right-aligned) ---
        pad = 8
        best_surf = font.render(f"Best: {self.viewer_best}", True, WHITE)
        last_surf = font.render(f"Last Best: {self.last_best}", True, WHITE)

        self.display.blit(best_surf, (self.w - best_surf.get_width() - pad, 0))
        self.display.blit(last_surf, (self.w - last_surf.get_width() - pad, 22))

        pygame.display.flip()

    def _move(self, action):
        """
        Updates the snake's direction and head position based on the action.
        :param action: A list representing the action [straight, right, left]
        """
        # [1, 0, 0] -> straight
        # [0, 1, 0] -> right turn
        # [0, 0, 1] -> left turn

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)