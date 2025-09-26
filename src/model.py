# model.py

import tensorflow as tf
import os

@tf.keras.utils.register_keras_serializable()
class Linear_QNet(tf.keras.Model):
    def __init__(self, hidden_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Create the layers in __init__ now, as we know the parameters
        self.linear1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.linear2 = tf.keras.layers.Dense(self.output_size)

    def call(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def get_config(self):
        # Return a dictionary of parameters needed to reconstruct the model
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        })
        return config

# The QTrainer class does not need to be changed
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function(jit_compile=True)   # << compile to graph (and XLA if available)
    def train_step(self, state, action, reward, next_state, done):
        # state: [B, 11], action: [B, 3] one-hot, reward: [B], next_state: [B, 11], done: [B]
        state      = tf.cast(state, tf.float32)
        next_state = tf.cast(next_state, tf.float32)
        action     = tf.cast(action, tf.float32)
        reward     = tf.cast(reward, tf.float32)
        done       = tf.cast(done, tf.float32)  # 1.0 if done, else 0.0

        with tf.GradientTape() as tape:
            q_pred = self.model(state)                       # [B, 3]
            q_next = self.model(next_state)                  # [B, 3]
            max_next = tf.reduce_max(q_next, axis=1)         # [B]
            q_target_action = reward + (1.0 - done) * self.gamma * max_next  # [B]

            # gather Q(s,a) via one-hot
            q_sa = tf.reduce_sum(q_pred * action, axis=1)    # [B]

            loss = self.loss_fn(q_target_action, q_sa)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))