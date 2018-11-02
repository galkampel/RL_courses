import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    with U.make_session(8):
        # Create the environment
        env = gym.make("CartPole-v0")
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()
		target_updates = [1,50,250,10000]
		num_episodes = 1000
		x_s = np.arange(num_episodes)
		y_s = np.zeros((len(target_updates),num_episodes))
		for i,target_update in enumerate(target_updates):
			episode_rewards = 0
			for j in range(num_episodes):
				obs = env.reset()
				for t in itertools.count(1):
					# Take action and update exploration to the newest value
					action = act(obs[None], update_eps=exploration.value(t))[0]
					new_obs, rew, done, _ = env.step(action)
					# Store transition in the replay buffer.
					replay_buffer.add(obs, action, rew, new_obs, float(done))
					obs = new_obs

					episode_rewards += rew
					if done:
						env.render()
						obs = env.reset()
						y_s[i,j] = episode_rewards
						break
				
					# Minimize the error in Bellman's equation on a batch sampled from replay buffer.
					if t > 1000:
						obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32) #change to dynamic
						train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))				

					#is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
					#if is_solved:
						# Show off the result
						#env.render()
					#else:
						# Minimize the error in Bellman's equation on a batch sampled from replay buffer.
						#if t > 1000:
							#obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32) #change to dynamic
							#train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
						# Update target network periodically.
						if t % target_update == 0:
							update_target()

					#if done and len(episode_rewards) % 10 == 0:
						#logger.record_tabular("steps", t)
						#logger.record_tabular("episodes", len(episode_rewards))
						#logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
						#logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
						#logger.dump_tabular()
		
		
		
		labels = ['Update target network after {} iterations'.format(target_update) for target_update in target_updates]
		colors = cm.Pastel1(list(range(len(target_updates))))
		for i,accReward in enumerate(accRewards):
			plt.plot(x_s,y_s[i,:],label = labels[i],color=colors[i])
		plt.grid()
		plt.title("Cumulative Rewards for different epsilons")
		plt.ylabel("Cumulative Discounted Rewards")
		plt.xlabel("Episode")
		plt.legend(loc = 'lower right')
		plt.savefig("plot1.png")
		plt.close()
		file = drive.CreateFile({'parents': [{"kind": "drive#childList",'id': '1zO5uSQKYGAUT4ZKi5rXmorP-vwQ9uG5d'}]})
        file.SetContentFile('plot1.png')
        file.Upload()
