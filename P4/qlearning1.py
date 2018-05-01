# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

import matplotlib
import matplotlib.pyplot as plt

learning_rate = .1
discount_factor = 0.9
screen_width  = 600
binsize = 50
screen_height = 400
vstates = 7
velocity_binsize = 20
num_actions = 2

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

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
    def random_action(self, p):
        return int(npr.rand() < p)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        d_gap = int(state['tree']['dist'] / binsize)
        v_gap = int((state['tree']['top'] - state['monkey']['top']) / binsize)
        vel = int(state['monkey']['vel'] / velocity_binsize)
        
        if vel < 0:
            vel = int(max(vel, -3))
        
        else:
            vel = int(min(vel, 3))
        
        action = self.random_action(.5)
        
        if self.last_action != None:
            last_d_gap = int(self.last_state['tree']['dist'] / binsize)
            last_v_gap = int((self.last_state['tree']['top'] - self.last_state['monkey']['top']) / binsize)
            last_vel = int(self.last_state['monkey']['vel'] / velocity_binsize)
            
            if last_vel < 0:
                last_vel = max(last_vel, -3)
            
            else:
                last_vel = min(last_vel, 3)
            
            action = int(self.Q[1][d_gap,v_gap,vel] > self.Q[0][d_gap,v_gap,vel])
            max_Q = self.Q[action][d_gap, v_gap, vel]
            self.Q[self.last_action][last_d_gap, last_v_gap, last_vel] += learning_rate*(self.last_reward + discount_factor * max_Q- self.Q[self.last_action][last_d_gap, last_v_gap, last_vel])
        
        self.last_action = action
        self.last_state = state
        return action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 1000, t_len = 100):
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
    axes[0].set_title('Scores Over Time\nQlearning 1')
    axes[0].set_xlabel('Num Iterations')
    axes[0].set_ylabel('Score')

    axes[1].hist(hist)
    axes[1].set_title('Score Distribution\nQlearning 1')
    axes[1].set_xlabel('Times Score Achieved')

    plt.tight_layout()
    plt.savefig('qlearning1_graphs.png')
    plt.clf()

    # Save history. 
    np.save('hist',np.array(hist))


