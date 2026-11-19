from gridworld import GridWorld
import numpy as np
import pandas as pd

# instantiate the gridworld object
world = GridWorld(size=5, 
                  state_includes_location=False, 
                  state_includes_sensory=True,  # this agent has features describing the state
                  randomize_each_trial=False
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

for step in range(5000):
    feature_vector = []
    action_utilities = []

    new_features, reward = world.step(action)
    new_feature_vector = np.array(list(new_features.values()) + 1)

    # choose an action
    # code here

# update weights
best_Q_new_features = weights.dot(new_feature_vector).max()
Q_for_executed_action = action_utilities[action]
rpe = (reward + gamma)

# As in task 1, visualize the agent navigating the world after it has finished learning. 
# Why can this agent handle the re-randomizing world better than the Q-learning agent from task 1?




# Why would a Q-learning agent take much longer to learn from these features, than from the 
# state in task 1? Why does function approximation not suffer from the same problem?
