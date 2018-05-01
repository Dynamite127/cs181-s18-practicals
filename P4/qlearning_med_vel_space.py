# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

import matplotlib
import matplotlib.pyplot as plt

discount_factor = 0.9
screen_width  = 600
binsize = 50
screen_height = 400
vstates = 17
# velocity_binsize = 8
num_actions = 2
epsilon = 0.1

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
        # we initialize the Q matrix for Q learning
        self.Q = np.zeros(
            (int(num_actions), 
             int(screen_width/binsize + 1), 
             int(screen_height/binsize + 1), 
             int(vstates))
        )
        
        # we count the number of times each state has been explored so that we can update epsilon to 0. 
        # intuitively, if we are perfectly learned, we do not need any more exploration.
        self.trials = np.zeros(
            (int(num_actions), 
             int(screen_width/binsize + 1), 
             int(screen_height/binsize + 1), 
             int(vstates))
        )

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
    def exploration(self, p):
        return int(npr.rand() < p)

    def discretize_velocity(self, velocity):
        # If velocity is less than -30, assign velocity to lowest state
        if velocity <= -30:
            return 0
        elif -29 <= velocity <= -26:
            return 1
        elif -25 <= velocity <= -22:
            return 2
        elif -21 <= velocity <= -18:
            return 3
        elif -17 <= velocity <= -14:
            return 4
        elif -13 <= velocity <= -10:
            return 5
        elif -9 <= velocity <= -6:
            return 6
        elif -5 <= velocity <= -2:
            return 7
        elif -1 <= velocity <= 2:
            return 8
        elif 3 <= velocity <= 6:
            return 9
        elif 7 <= velocity <= 10:
            return 10
        elif 11 <= velocity <= 14:
            return 11
        elif 15 <= velocity <= 18:
            return 12
        elif 19 <= velocity <= 22:
            return 13
        elif 23 <= velocity <= 25:
            return 14
        elif 26 <= velocity <= 29:
            return 15
        elif velocity >= 30:
            return 16
        else:
            assert(0)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # Get data from current state
        d_gap = int(state['tree']['dist'] / binsize)
        v_gap = int((state['tree']['top'] - state['monkey']['top']) / binsize)
        # vel = int(state['monkey']['vel'] / velocity_binsize)
        vel = self.discretize_velocity(state['monkey']['vel'])
        
        # # If velocity too high or low, place into appropriate bin
        # if np.abs(vel) > 8:
        #     vel = int(8 * np.sign(vel))
        
        action = self.exploration(.5)
        
        if self.last_action != None:
            last_d_gap = int(self.last_state['tree']['dist'] / binsize)
            last_v_gap = int((self.last_state['tree']['top'] - self.last_state['monkey']['top']) / binsize)
            # last_vel = int(self.last_state['monkey']['vel'] / velocity_binsize)
            last_vel = self.discretize_velocity(self.last_state['monkey']['vel'])
            
            # # Compress velocity if needed
            # if np.abs(last_vel) > 8:
            #     last_vel = int(8 * np.sign(last_vel))
            
            # Max Q value over all actions for this particular distance from tree, vertical dist from tree,
            # velocity
            max_Q = np.max(self.Q[:, d_gap, v_gap, vel])
            new_epsilon = epsilon / max(self.trials[action][d_gap, v_gap, vel], 1)
            
            if npr.rand() > new_epsilon:
                if self.Q[1][d_gap, v_gap, vel] > self.Q[0][d_gap, v_gap, vel]:
                    action = 1
                else: 
                    action = 0

            # Learning rate decreases as number of times we execute the last action in the last state
            # increases
            eta = 1 / self.trials[self.last_action][last_d_gap, last_v_gap, last_vel]
            q_adjust = eta * (self.last_reward + discount_factor * max_Q - self.Q[self.last_action][last_d_gap, last_v_gap, last_vel])
            self.Q[self.last_action][last_d_gap, last_v_gap, last_vel] += q_adjust
        
        self.last_action = action
        self.last_state = state
        self.trials[action][d_gap, v_gap, vel] += 1

        return action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

def run_games(learner, hist, iters = 10000, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)
        print('{}: {}'.format((ii + 1),learner.last_state['score']))

        # Reset the state of the learner.
        learner.reset()

        
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. 
    num_iters = 2000
    time_step = 2
    try:
        run_games(agent, hist, num_iters, time_step)
    except:
        pass

    fig, axes = plt.subplots(2, 1, figsize = (10, 8))

    axes[0].plot(range(len(hist)), hist, 'o')
    axes[0].set_title('Scores Over Time\nMed Velocity Space')
    axes[0].set_xlabel('Num Iterations')
    axes[0].set_ylabel('Score')

    axes[1].hist(hist)
    axes[1].set_title('Score Distribution\nMed Velocity Space')
    axes[1].set_xlabel('Times Score Achieved')

    plt.tight_layout()
    plt.savefig('med_vel_space_graphs.png')
    plt.clf()

    # Save history. 
    np.save('hist',np.array(hist))

