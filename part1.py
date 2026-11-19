from gridworld import GridWorld
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# instantiate the gridworld object
world = GridWorld(size=5, 
                  state_includes_location=True, 
                  state_includes_sensory=False, 
                  randomize_each_trial=False
                  )


# do Q learning here
# plot the cumulative reward vs. step #
# and after learning, visualize the agent's behavior in the world

state = world.reset()
actions = world.get_actions(state)
print(state)
Q = pd.DataFrame()
gamma = 0.9
alpha = 0.1
epsilon = 0.1 # Do a random action randomly

# For plotting graph
rewards = []
num_steps = []

# convert states into strings because they're hashable
for step in range(3000):
    if str(state) not in Q.index:
        Q.loc[str(state), world.get_actions()] = 1 # try changing me!

    if np.random.random() < epsilon:
        action = np.random.choice(actions)
    else:
        action = Q.loc[str(state)].idxmax()
    new_state, reward  = world.step(action)
    rewards.append(reward)
    num_steps.append(step)

    if str(new_state) not in Q.index:
        Q.loc[str(new_state), world.get_actions()] = 1

    # td_error ← r + 𝛾•maxa’(Q[s’, a’]) - Q[s, a]
    rpe = (reward + gamma * Q.loc[str(new_state)].max()) - Q.loc[str(state), action]
    Q.loc[str(state), action] += alpha * rpe

    state = new_state
    #world.render()
    
    #world.reset()
    # Resetting the world makes the agent unable to learn because the world keeps changing and its unable to figure out which states are beneficial
print(Q)

plt.plot(num_steps, np.cumsum(rewards))

plt.xlabel("reward")
plt.ylabel("# of steps")

plt.show()


# Now set the world to re-randomize after each trial. What goes wrong? Why?