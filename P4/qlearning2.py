# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

import matplotlib
import matplotlib.pyplot as plt

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.discount_factor = 0.9
        self.screen_width  = 600
        self.binsize = 50
        self.screen_height = 400
        self.velocity_binsize = 20
        self.vstates = 5
        self.num_actions = 2
        self.epsilon = 0.001
        
        # we initialize the Q matrix for Q learning
        self.Q = np.zeros(
            (int(self.num_actions), 
             int(self.screen_width/self.binsize + 1), 
             int(self.screen_height/self.binsize + 1), 
             int(self.vstates))
        )
        
        # we count the number of times each state has been explored so that we can update self.epsilon to 0. 
        # intuitively, if we are perfectly learned, we do not need any more exploration.
        self.trials = np.zeros(self.Q.shape)

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        
    def exploration(self, p):
        return int(npr.rand() < p)

    def discretize_velocity(self, state):
        vel = int(state['monkey']['vel'] / self.velocity_binsize)
        if vel <= -2:
            vel = 0
        elif vel == -1:
            vel = 1
        elif vel == 0:
            vel = 2
        elif vel == 1:
            vel = 3
        elif vel >= 2:
            vel = 4
        else:
            assert(0)
        return vel
    
    def get_feats(self, state):
        d_gap = int(state['tree']['dist'] / self.binsize)
        v_gap = int((state['tree']['top'] - state['monkey']['top']) / self.binsize)
        vel = self.discretize_velocity(state)
        return tuple([d_gap, v_gap, vel])

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # Get data from current state
        feats = self.get_feats(state)
        action = self.exploration(.5)
        
        if self.last_action != None:
            last_feats = self.get_feats(self.last_state)
            
            # Max Q value over all actions for this particular distance from tree, vertical dist from tree,
            # velocity
            new_epsilon = self.epsilon / max(self.trials[action][feats], 1)
            
            if npr.rand() > new_epsilon:
                action = int(self.Q[1][feats] > self.Q[0][feats])

            max_Q = self.Q[action][feats]

            # Learning rate decreases as number of times we execute the last action in the last state
            # increases
            eta = 1 / self.trials[self.last_action][last_feats]
            q_adjust = eta * (self.last_reward + self.discount_factor * max_Q - self.Q[self.last_action][last_feats])
            self.Q[self.last_action][last_feats] += q_adjust
        
        self.last_action = action
        self.last_state = state
        self.trials[action][feats] += 1

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
    num_iters = 1000
    time_step = 2
    run_games(agent, hist, num_iters, time_step)

    # Calculate Running Avg
    avgs_lst = []
    for i in range(0, len(hist)):
        n = i + 1
        if i == 0:
            avgs_lst.append(
                round(hist[i], 2)
            )
        else:
            avgs_lst.append(
                round(1.0 * avgs_lst[i - 1] * ((n - 1) / n ) + 1.0 * (hist[i] / n), 2)
            )
        

    fig, axes = plt.subplots(3, 1, figsize = (10, 8))

    axes[0].plot(range(len(hist)), hist, 'o')
    axes[0].set_title('Scores Over Time\nQlearning 2')
    axes[0].set_xlabel('Num Iterations')
    axes[0].set_ylabel('Score')

    axes[1].hist(hist)
    axes[1].set_title('Score Distribution\nQlearning 2')
    axes[1].set_xlabel('Times Score Achieved')

    axes[2].plot(range(1, len(avgs_lst) + 1), avgs_lst)
    axes[2].set_title('Running Avg of Scores Over Time')
    axes[2].set_xlabel('Num Iterations')
    axes[2].set_ylabel('Avg Score')

    plt.tight_layout()
    plt.savefig('qlearning2_graphs.png')
    plt.clf()

    # Save history. 
    np.save('hist',np.array(hist))

