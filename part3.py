from gridworld import GridWorld
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# instantiate the gridworld object
world = GridWorld(size=5, 
                  state_includes_location=False, 
                  state_includes_sensory=True,  # this agent has features describing the state
                  randomize_each_trial=False,
                  wall_probability=0.1
                  )
features = world.reset()

# Write an agent that uses function approximation to solve this world

# create a matrix of weights for each feature and action
weights = pd.DataFrame(index=['up', 'down', 'left', 'right'], 
                       columns=list(features.keys()) + [ 'bias'], 
                       data=np.random.random(size=(4, len(features)+1))
                       )

# learning parameters
gamma = 0.9
alpha = 0.1
epsilon = 0.1 # Do a random action randomly
all_reward = []
num_steps = []

rng = np.random.default_rng()

for step in range(5000):
    feature_vector = []
    action_utilities = []
    features = world.get_current_state()

    feature_vector = np.array(list(features.values()) + [1]) # + 1 for the bias column
    # dot product the weights matrix with vector of features.
    action_utilities = weights.dot(feature_vector)

    # e-greedy algorhithm
    if (epsilon > rng.random()):
        action = rng.choice(weights.index)
    else:
        # choose action. The action with the highest utiltity calculated with a linear model.
        action = action_utilities.idxmax()

    # Take the action and get info on new state
    new_features, reward = world.step(action)

    # For plotting
    all_reward.append(reward)
    num_steps.append(step)
    
    new_feature_vector = np.array(list(new_features.values()) + [1])

    # update weights
    best_Q_new_features = weights.dot(new_feature_vector).max()
    Q_for_executed_action = action_utilities[action]
    rpe = (reward + gamma * best_Q_new_features) - Q_for_executed_action
    # Tune weights to minimize square-RPE
    weights.loc[action] += alpha * rpe * feature_vector

    #world.render()

# As in task 1, visualize the agent navigating the world after it has finished learning. 
# Why can this agent handle the re-randomizing world better than the Q-learning agent from task 1?

plt.plot(num_steps, np.cumsum(all_reward))

plt.xlabel("# of steps")
plt.ylabel("reward")

plt.show()
# A function approximation agent can handle a new world better because it acts based off features in its
# environment. If it learned the red square is bad in a previous world, it will retain that 
# knowledge in the new random world. A Q-learning agent doesn't do as well since it has to learn all
# of a world's states individually. When a new world is created, all the previous world's knowledge becomes 
# useless.

# Why would a Q-learning agent take much longer to learn from these features, than from the 
# state in task 1? Why does function approximation not suffer from the same problem?

# The Q-learning agent would take much longer to learn from these features because it would have to 
# store each unique combination of features as its own unique state. This requires the agent to learn 
# new Q values for each (state, action) pairs, even if it had previously learned from similar states.
# Function approximation doesn't suffer from this because it instead uses generalization in its approach 
# by learning weights for the features. These weights can then be applied to any state containing similar features,
# allowing the agent to use what it has already learned and therefore reducing the time it takes for 
# the agent to learn overall.