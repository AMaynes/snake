# Snake AI with Reinforcement Learning 🐍🧠

This project implements an autonomous agent that learns to play the classic game of Snake using a Deep Q-Network (DQN), a form of Reinforcement Learning. The agent learns from scratch through trial and error, with the goal of maximizing its score.

The entire project is built with Python, using Pygame for the game environment and TensorFlow/Keras for the neural network.



## How It Works

The intelligence of the agent is not hard-coded. Instead, it's an **emergent behavior** that results from a simple feedback loop. The agent's goal is to take actions that maximize its future rewards.

---
### The Core Components

The system is built around three main components: the **Environment**, the **Agent**, and the **Model**.

#### 1. The Environment (`snake_game.py`)
This is the world where the agent lives and learns. It's a complete implementation of the Snake game built with Pygame. Its most important job is to:
* Provide the agent with its current **state**.
* Receive an **action** from the agent.
* Update the game and return a **reward** based on the outcome of that action.

---
#### 2. The Agent (`agent.py`)

The agent is the decision-maker. It observes the state from the environment and decides which action to take. This is guided by the principles of Reinforcement Learning.

* **State:** The agent doesn't "see" the screen. It receives a simplified 11-element vector that describes its immediate situation.
    * `[Danger Straight, Danger Right, Danger Left]` (3 values)
    * `[Current Direction: L, R, U, D]` (4 values)
    * `[Food Location: L, R, U, D]` (4 values)
    

* **Action:** At any point, the snake can only make three moves relative to its current direction: `[Go Straight, Turn Right, Turn Left]`. The agent's job is to choose one of these three actions.

* **Reward:** The agent receives feedback from the environment after each move:
    * **+10** for eating food.
    * **-100** for dying.
    * *A small positive/negative reward for moving toward/away from the food (optional reward shaping).*

---
#### 3. The Neural Network (`model.py`)

The neural network is the agent's "brain." It's a Deep Q-Network whose job is to predict the expected future reward for each of the three possible actions, given the current state.

**The Architecture:**
* **Input Layer:** 11 neurons (for the state vector).
* **Hidden Layers:** Two layers with 256 neurons each, using the ReLU activation function. These layers find complex patterns in the state data.
* **Output Layer:** 3 neurons, one for each possible action. The values they output are the predicted **Q-values**.



The agent simply performs the action corresponding to the neuron with the highest Q-value.

---
### The Training Loop (`train.py`)

The learning happens in a continuous feedback loop driven by the `train.py` script.



1.  **Get State:** The agent gets the current state from the game.
2.  **Get Action:** The agent feeds the state into its neural network to get Q-values and chooses the best action. (Initially, it chooses random actions to explore).
3.  **Perform Action:** The agent sends the chosen action to the game, which updates.
4.  **Get Feedback:** The game returns the new state, the reward, and whether the game is over.
5.  **Store & Train:** The agent stores this entire experience (`state`, `action`, `reward`, `next_state`) in its memory. It then trains its neural network on a random batch of past experiences, which slowly improves the accuracy of its Q-value predictions.

Over thousands of games, this loop adjusts the network's weights until it becomes an expert at predicting moves that lead to high scores.

---
## Project Structure
```
snake-ai/
├── model/
│   ├── model.keras       (Saved model)
│   └── record.txt        (Saved high score)
├── agent.py              (The RL agent and decision logic)
├── constants.py          (Game colors, speed, etc.)
├── model.py              (The Keras neural network model)
├── snake_game.py         (The Pygame environment)
└── train.py              (The main script to run for training)
```

---
## Setup and Usage

### Prerequisites
* Python 3.9+
* An NVIDIA GPU is recommended for faster training but not required.

### 1. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

```bash
# 1. Create the environment
python -m venv venv

# Activate it
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# 2. Install Dependencies
pip install tensorflow pygame numpy

# 3. Run the Training
python src/train.py

# 4. Watch the Agent Play