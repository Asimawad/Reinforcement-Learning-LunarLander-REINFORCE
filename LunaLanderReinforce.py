from shutil import rmtree # deleting directories
import random
import pickle
import collections # useful data structures
import numpy as np
from gymnasium.wrappers import RecordVideo
import jax
import jax.numpy as jnp # jax numpy
import haiku as hk # jax neural network library
import optax # jax optimizer library
import matplotlib.pyplot as plt # graph plotting library
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym

# Create the environment
env_name = "LunarLander-v2"  
env = gym.make(env_name)

# Reset the environment
s_0 = env.reset()[0]
print("Initial State::", s_0)

# Get environment obs space
obs_shape = env.observation_space.shape
print("Environment Obs Space Shape:", obs_shape)

# Get action space - e.g. discrete or continuous
print(f"Environment action space: {env.action_space}")

# Get num actions
num_actions = env.action_space.n
print(f"Number of actions: {num_actions}")


# Function to save logs to a file
def save_logs(logs, file_name="training_logs.txt"):
    with open(file_name, "a") as log_file:  # "a" mode to append logs if the file already exists
        log_file.write("\n".join(logs) + "\n")

# Function to save model parameters
def save_model_parameters(params, file_name="learned_params.pkl"):
    with open(file_name, "wb") as param_file:
        pickle.dump(params, param_file)

# Function to save the episode returns list
def save_episode_returns(returns_list, file_name="episode_returns.pkl"):
    with open(file_name, "wb") as file:
        pickle.dump(returns_list, file)
def compute_weighted_log_prob(action_prob, episode_return):

    log_porb = jnp.log(action_prob)

    weighted_log_prob = episode_return * log_porb

    return weighted_log_prob

def compute_returns(rewards, gamma=0.99):
  returns = []
  for i in range(len(rewards)):
      G = 0
      for j in range(i, len(rewards)):
          G += (gamma**(j-i))*rewards[j]
      returns.append(G)
  return returns

# NamedTuple to store memory
EpisodeReturnsMemory = collections.namedtuple("EpisodeReturnsMemory", ["obs", "action", "returns"])

class EpisodeReturnsBuffer:

    def __init__(self, num_transitions_to_store=512, batch_size=256):
        self.batch_size = batch_size
        self.memory_buffer = collections.deque(maxlen=num_transitions_to_store)
        self.current_episode_transition_buffer = []

    def push(self, transition):
        self.current_episode_transition_buffer.append(transition)
        done = transition.terminated or transition.truncated
        if done:
            episode_rewards = []
            for t in self.current_episode_transition_buffer:
                episode_rewards.append(t.reward)

            G = compute_returns(episode_rewards)

            for i, t in enumerate(self.current_episode_transition_buffer):
                memory = EpisodeReturnsMemory(t.obs, t.action, G[i])
                self.memory_buffer.append(memory)

            # Reset episode buffer
            self.current_episode_transition_buffer = []


    def is_ready(self):
        return len(self.memory_buffer) >= self.batch_size

    def sample(self):
        random_memory_sample = random.sample(self.memory_buffer, self.batch_size)

        obs_batch, action_batch, returns_batch = zip(*random_memory_sample)

        return EpisodeReturnsMemory(
            np.stack(obs_batch).astype("float32"),
            np.asarray(action_batch).astype("int32"),
            np.asarray(returns_batch).astype("int32")
        )
# Instantiate Memory
REINFORCE_memory = EpisodeReturnsBuffer(num_transitions_to_store=512, batch_size=128)

def make_policy_network(num_actions: int, layers=[20, 20]) -> hk.Transformed:
  """Factory for a simple MLP network for the policy."""

  def policy_network(obs):
    network = hk.Sequential(
        [
            hk.Flatten(),
            hk.nets.MLP(layers + [num_actions])
        ]
    )
    return network(obs)

  return hk.without_apply_rng(hk.transform(policy_network))

# Example
POLICY_NETWORK = make_policy_network(num_actions = num_actions, layers=[28,28])
random_key = jax.random.PRNGKey(0) # random key
dummy_obs = np.ones(obs_shape, "float32")

# Initialise parameters
REINFORCE_params = POLICY_NETWORK.init(random_key, dummy_obs)
print("Initial params:", REINFORCE_params.keys())

# Pass input through the network
output = POLICY_NETWORK.apply(REINFORCE_params, dummy_obs)
print("Policy network output:", output)

def sample_action(random_key, logits):
    return jax.random.categorical(random_key,logits)


def REINFORCE_choose_action(key, params, actor_state, obs, evaluation=False):
  obs = jnp.expand_dims(obs, axis=0) # add dummy batch dim before passing through network

  # Pass obs through policy network to compute logits
  logits = POLICY_NETWORK.apply(params, obs)
  logits = logits[0] # remove batch dim

  # Randomly sample action
  sampled_action = sample_action(key, logits)

  return sampled_action, actor_state


def policy_gradient_loss(action, logits, returns):

  # YOUR CODE
# normalize
  all_action_probs = jax.nn.softmax(logits) # convert logits into probs

  action_prob = all_action_probs[action]

  weighted_log_prob = compute_weighted_log_prob(action_prob, returns)

  # END YOUR CODE

  loss = - weighted_log_prob # negative because we want gradient `ascent`
# 1620
  return loss
def batched_policy_gradient_loss(params, obs_batch, action_batch, returns_batch):
    # Get logits by passing observation through network
    logits_batch = POLICY_NETWORK.apply(params, obs_batch)

    policy_gradient_loss_batch = jax.vmap(policy_gradient_loss)(action_batch, logits_batch, returns_batch) # add batch

    # Compute mean loss over batch
    mean_policy_gradient_loss = jnp.mean(policy_gradient_loss_batch)

    return mean_policy_gradient_loss

REINFORCE_OPTIMIZER = optax.adam(0.0001)

# Initialise the optimiser
REINFORCE_optim_state = REINFORCE_OPTIMIZER.init(REINFORCE_params)

# A NamedTuple to store the state of the optimiser
REINFORCELearnState = collections.namedtuple("LearnerState", ["optim_state"])

def REINFORCE_learn(key, params, learner_state, memory):

  # Get the policy gradient by using `jax.grad()` on `batched_policy_gradient_loss`
  grad_loss = jax.grad(batched_policy_gradient_loss)(params, memory.obs, memory.action, memory.returns)

  # Get param updates using gradient and optimizer
  updates, new_optim_state = REINFORCE_OPTIMIZER.update(grad_loss, learner_state.optim_state)

  # Apply updates to params
  params = optax.apply_updates(params, updates)

  return params, REINFORCELearnState(new_optim_state) # update learner state

# NamedTuple to store transitions
Transition = collections.namedtuple("Transition", ["obs", "action", "reward", "next_obs", "terminated", "truncated"])
# Training Loop
def run_training_loop(env_name, agent_params, agent_select_action_func,
    agent_actor_state=None, agent_learn_func=None, agent_learner_state=None,
    agent_memory=None, num_episodes=1000, evaluator_period=100,
    evaluation_episodes=10, learn_steps_per_episode=1,
    train_every_timestep=False, video_subdir="",):
   
    # Setup Cartpole environment and recorder
    env = gym.make(env_name, render_mode="rgb_array")        # training environment
    eval_env = gym.make(env_name, render_mode="rgb_array")   # evaluation environment

    # Video dir
    video_dir = "./video"+"/"+video_subdir

    # Clear video dir
    try:
      rmtree(video_dir)
    except:
      pass

    # Wrap in recorder
    env = RecordVideo(env, video_dir+"/train", episode_trigger=lambda x: (x % evaluator_period) == 0, disable_logger= True)
    eval_env = RecordVideo(eval_env, video_dir+"/eval", episode_trigger=lambda x: (x % evaluation_episodes) == 0,disable_logger=True)

    # JAX random number generator
    rng = hk.PRNGSequence(jax.random.PRNGKey(0))
    random.seed(0)

    episode_returns = []                     # List to store history of episode returns.
    evaluator_episode_returns = []           # List to store history of evaluator returns.
    timesteps = 0
    for episode in range(num_episodes):

        # Reset environment.
        obs = env.reset()[0]  # new way to seed the environment

        episode_return = 0
        done = False

        while not done:

            # Agent select action.
            action, agent_actor_state = agent_select_action_func(
                                            next(rng),
                                            agent_params,
                                            agent_actor_state,
                                            np.array(obs)
                                        )

            # Step environment.
            next_obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            # Pack into transition.
            transition = Transition(obs, action, reward, next_obs, terminated, truncated)

            # Add transition to memory.
            if agent_memory: # check if agent has memory
              agent_memory.push(transition)

            # Add reward to episode return.
            episode_return += reward 
            # Set obs to next obs before next environment step. CRITICAL!!!
            obs = next_obs

            # Increment timestep counter
            timesteps += 1

            # Maybe learn every timestep
            if train_every_timestep and (timesteps % 4 == 0) and agent_memory and agent_memory.is_ready(): # Make sure memory is ready
                # First sample memory and then pass the result to the learn function
                memory = agent_memory.sample()
                agent_params, agent_learner_state = agent_learn_func(
                                                        next(rng),
                                                        agent_params,
                                                        agent_learner_state,
                                                        memory
                                                    )# Ma fihimta

        episode_returns.append(episode_return)

        # At the end of every episode we do a learn step.
        if agent_memory and agent_memory.is_ready(): # Make sure memory is ready

            for _ in range(learn_steps_per_episode):
                # First sample memory and then pass the result to the learn function
                memory = agent_memory.sample()
                agent_params, agent_learner_state = agent_learn_func(
                                                        next(rng),
                                                        agent_params,
                                                        agent_learner_state,
                                                        memory
                                                    )

        if (episode % evaluator_period) == 0: # Do evaluation

            evaluator_episode_return = 0
            for eval_episode in range(evaluation_episodes):
                obs = eval_env.reset()[0]
                done = False
                while not done:
                    action, _ = agent_select_action_func(
                                    next(rng),
                                    agent_params,
                                    agent_actor_state,
                                    np.array(obs),
                                    evaluation=True
                                )

                    obs, reward, terminated, truncated, _ = eval_env.step(int(action))
                    done = terminated or truncated

                    evaluator_episode_return += reward

            evaluator_episode_return /= evaluation_episodes

            evaluator_episode_returns.append(evaluator_episode_return)

            logs = [
                    f"Episode: {episode}",
                    f"Episode Return: {episode_return}",
                    f"Average Episode Return: {np.mean(episode_returns[-20:])}",
                    f"Evaluator Episode Return: {evaluator_episode_return}"
            ]

            print(*logs, sep="\t") # Print the logs

    env.close()
    eval_env.close()

    return episode_returns, evaluator_episode_returns
# env.seed
# JIT the choose_action and learn functions for more speed
REINFORCE_learn_jit = jax.jit(REINFORCE_learn)
REINFORCE_choose_action_jit = jax.jit(REINFORCE_choose_action)
# Initial learn state
REINFORCE_learn_state = REINFORCELearnState(REINFORCE_optim_state)

# Run training loop
print("Starting training. This may take a few minutes to complete.")
episode_returns, evaluator_returns = run_training_loop(
                                        env_name,
                                        REINFORCE_params,
                                        REINFORCE_choose_action_jit,
                                        None, # action state not used
                                        REINFORCE_learn_jit,
                                        REINFORCE_learn_state,
                                        REINFORCE_memory,
                                        num_episodes=25001,
                                        learn_steps_per_episode = 4,     # this has changed
                                        video_subdir="Lunalanderreinforce"
                                      )

save_model_parameters(REINFORCE_params, "./ass2b/learned_params.pkl")
save_episode_returns(episode_returns, file_name="./ass2b/episode_returns.pkl")

# Plot the episode returns
plt.plot(range(len(episode_returns)), episode_returns)  # Corrected the plot call
plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.title("REINFORCE")
plt.savefig('./ass2b/llreinforce_loss.pdf')
plt.show()
# started training at 14:12.