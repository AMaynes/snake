# train.py
import os
# Make sure we can show a window (remove dummy driver if you had it)
os.environ.pop("SDL_VIDEODRIVER", None)

from agent import Agent
from snake_game import SnakeGameAI
from constants import *
import numpy as np
import tensorflow as tf

NUM_HEADLESS = 20          # number of extra envs (start with 4–16)
SAVE_EVERY_GAMES = 100    # your existing cadence
MODEL_PATH = "./model/model.keras"
GAMES_COUNT_PATH = "./model/games_count.txt"

def train():
    record = 0
    # load record if present
    try:
        with open('./model/record.txt') as f:
            record = int((f.read() or '0').strip())
    except FileNotFoundError:
        pass

    agent = Agent(verbose=True)  # prints once at startup
    # 1) The visible env (main learner) in the main thread
    viewer = SnakeGameAI(render=True, render_every=1)  # render every N frames
    last_mtime = 0.0
    viewer_model = None

    # 2) Headless envs
    envs = [SnakeGameAI(render=False) for _ in range(NUM_HEADLESS)]

    while True:
        # ----- epsilon by total games played
        eps = max(0, 80 - agent.n_games)

        # ----- (A) VIEWER: reload newer model if present, and pick action
        # (check file modification time so we only reload/print when truly newer)
        if os.path.exists(MODEL_PATH):
            mtime = os.path.getmtime(MODEL_PATH)
            if mtime > last_mtime:
                try:
                    viewer_model = tf.keras.models.load_model(MODEL_PATH)
                    last_mtime = mtime
                    games_played = None
                    try:
                        with open(GAMES_COUNT_PATH) as f:
                            games_played = f.read().strip()
                    except FileNotFoundError:
                        pass
                    print(f"[Viewer] Reloaded model"
                        f"{f' at game {games_played}' if games_played else ''}.")
                except Exception as e:
                    print(f"[Viewer] Failed to reload model: {e}")

        # decide viewer action:
        viewer_state = agent.get_state(viewer)
        if np.random.randint(0, 201) < eps:
            v_move = np.random.randint(0, 3)
        else:
            # prefer the latest on-disk model if loaded; otherwise use live learner
            model_for_view = viewer_model if viewer_model is not None else agent.model
            v_q = model_for_view(tf.convert_to_tensor(viewer_state[None, :], dtype=tf.float32))
            v_move = int(tf.argmax(v_q[0]).numpy())
        viewer_action = [0, 0, 0]; viewer_action[v_move] = 1

        # ----- (B) HEADLESS: batch predict actions for headless envs using live learner
        headless_states = [agent.get_state(g) for g in envs]
        headless_states_np = np.asarray(headless_states, dtype=np.float32)
        hq = agent.model(tf.convert_to_tensor(headless_states_np))
        headless_actions = []
        for i in range(len(envs)):
            if np.random.randint(0, 201) < eps:
                m = np.random.randint(0, 3)
            else:
                m = int(tf.argmax(hq[i]).numpy())
            oh = [0, 0, 0]; oh[m] = 1
            headless_actions.append(oh)


        # ----- 4) Step viewer (renders)
        vr, vdone, vscore = viewer.play_step(viewer_action)
        vnext = agent.get_state(viewer)
        agent.remember(viewer_state, viewer_action, vr, vnext, vdone)
        agent.train_short_memory(viewer_state, viewer_action, vr, vnext, vdone)

        if vdone:
            viewer.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if vscore > record:
                record = vscore
                agent.model.save('./model/model.keras')
                with open('./model/record.txt', 'w') as f: f.write(str(record))
                print("--- New Record! Model Saved. ---")

            # cumulative counter (simple mirror; switch to atomic incr if you like)
            with open('./model/games_count.txt', 'w') as f:
                f.write(str(agent.n_games))

            if agent.n_games % SAVE_EVERY_GAMES == 0:
                agent.model.save('./model/model.keras')
                print(f"--- Periodic Save at Game {agent.n_games}. ---")

            print('Game', agent.n_games, 'Score', vscore, 'Record:', record)

        # ----- 5) Step headless envs
        for i, g in enumerate(envs, start=1):  # actions[i] belongs to envs[i-1]
            r, done, score = g.play_step(headless_actions[i-1])
            ns = agent.get_state(g)
            agent.remember(headless_states[i-1], headless_actions[i-1], r, ns, done)
            agent.train_short_memory(headless_states[i-1], headless_actions[i-1], r, ns, done)

            if done:
                g.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save('./model/model.keras')
                    with open('./model/record.txt', 'w') as f: f.write(str(record))
                    print("--- New Record! Model Saved. ---")

                with open('./model/games_count.txt', 'w') as f:
                    f.write(str(agent.n_games))

                if agent.n_games % SAVE_EVERY_GAMES == 0:
                    agent.model.save('./model/model.keras')
                    print(f"--- Periodic Save at Game {agent.n_games}. ---")

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

if __name__ == '__main__':
    train()
