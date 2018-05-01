# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

import matplotlib
import matplotlib.pyplot as plt

discount_factor = 0.9
screen_width  = 600
width_binsize = 40
screen_height = 400
height_binsize = 25
vstates = 5
velocity_binsize = 20
num_actions = 2
epsilon = 0.001

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
             int(screen_width/width_binsize + 1),
             int(screen_height/height_binsize + 1), 
             int(screen_height/height_binsize + 1), 
             int(vstates))
        )
        
        # we count the number of times each state has been explored so that we can update epsilon to 0. intuitively, if we are perfectly learned, we do not need any more exploration.
        self.trials = np.zeros(
            (int(num_actions), 
             int(screen_width/width_binsize + 1),
             int(screen_height/height_binsize + 1), 
             int(screen_height/height_binsize + 1), 
             int(vstates))
        )

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
    def exploration(self, p):
        return int(npr.rand() < p)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        d_gap = int(state['tree']['dist'] / width_binsize)
        v_gap = int((state['tree']['top']-state['monkey']['top']) / height_binsize)
        position = int(state['monkey']['top'] / height_binsize)
        vel = int(state['monkey']['vel'] / velocity_binsize)
        
        if np.abs(vel) > 2:
            vel = int(2 * np.sign(vel))
        action = int(self.exploration(.1))
        
        if self.last_action != None:
            last_d_gap = int(self.last_state['tree']['dist'] / width_binsize)
            last_v_gap = int((self.last_state['tree']['top'] - self.last_state['monkey']['top']) / height_binsize)
            last_position = int(self.last_state['monkey']['top'] / height_binsize)
            last_vel = int(self.last_state['monkey']['vel'] / velocity_binsize)
            
            if np.abs(last_vel) > 2:
                last_vel = int(2 * np.sign(last_vel))
            
            max_Q = np.max(self.Q[:,d_gap,v_gap,vel])
            new_epsilon = epsilon / max(self.trials[action][d_gap, v_gap, position, vel], 1)
            
            if (npr.rand() > new_epsilon):
                action = int(self.Q[1][d_gap,v_gap,position,vel] > self.Q[0][d_gap,v_gap,position,vel])
            
            eta = 1 / self.trials[self.last_action][last_d_gap, last_v_gap, last_position, last_vel]
            self.Q[self.last_action][last_d_gap, last_v_gap, last_position, last_vel] += eta*(self.last_reward + discount_factor * max_Q- self.Q[self.last_action][last_d_gap, last_v_gap, last_position, last_vel])
        
        self.last_action = action
        self.last_state = state
        self.trials[action][d_gap, v_gap, position, vel] += 1
        
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
    axes[0].set_title('Scores Over Time\nQlearning 3')
    axes[0].set_xlabel('Num Iterations')
    axes[0].set_ylabel('Score')

    axes[1].hist(hist)
    axes[1].set_title('Score Distribution\nQlearning 3')
    axes[1].set_xlabel('Times Score Achieved')

    plt.tight_layout()
    plt.savefig('qlearning3_graphs.png')
    plt.clf()

    # Save history. 
    np.save('hist',np.array(hist))

