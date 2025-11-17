# VALUE ITERATION ALGORITHM

## AIM
To find an optimal policy for an agent navigating a grid-world with slippery tiles, aiming to reach a goal state while maximizing expected rewards using value iteration algorithm.

## PROBLEM STATEMENT
The problem involves using the Value Iteration algorithm to find the best strategy for an agent in the Frozen Lake environment. The agent must navigate icy terrain, avoid hazards, and reach the goal while optimizing cumulative rewards in an uncertain environment.
## POLICY ITERATION ALGORITHM
Step 1:
Set the value of each state to 0 (initial guess).<br>
Step 2:
Look at all the actions you can take from that state (like moving up, down, left, or right).<br>
Step 3:
Calculate the expected value of each action (i.e., how good that action is based on its possible results).<br>
Step 4:
Pick the action that gives the highest value and update the value of the state with that number.<br>
Step 5:
Keep updating the values for all states until the difference between the old and new values is very small.<br>
Step 6:
Once the values have stabilized, go through each state again and pick the action that leads to the highest value. This gives you the optimal action (policy) for each state.<br>
### ENVIRONMENT : 
```python
envdesc  = ['FSHF','FHGH','FFHF', 'GFFH']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 12 #(Goal State)
P = env.env.P
```

## VALUE ITERATION FUNCTION
#### Name: MUKESH P
#### Register Number: 212222240068
```python
def value_iteration(P, gamma=1.0, theta=1e-10):
  V = np.zeros(len(P), dtype=np.float64)
  while True:
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state, reward, done in P[s][a]:
          Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
      break

    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    
    return V, pi
```
## OUTPUT:
#### Optimal policy and state-value function :
<img width="381" height="127" alt="41" src="https://github.com/user-attachments/assets/a0aff32b-d181-4e9e-8d4f-74c4237674ff" />


#### Pobability Success :
<img width="523" height="36" alt="42" src="https://github.com/user-attachments/assets/d26d18f8-1c14-491e-a0c7-242fb3efd51e" />


#### State Value Function :
<img width="387" height="98" alt="43" src="https://github.com/user-attachments/assets/d2d01730-1a16-4da7-89a2-f12238983a8e" />



## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
