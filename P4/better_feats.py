# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey
eta = 0.1
discount_factor = 0.9
screen_width  = 600
screen_height = 400
vstates = 5
num_actions = 2
epsilon = 0.001
num_feats = 8

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        # we initialize the Q matrix for Q learning
        self.Q = np.zeros(tuple([2 for _ in range(num_feats + 1)]))
        # we count the number of times each state has been explored so that we can update epsilon to 0. intuitively, if we are perfectly learned, we do not need any more exploration.
        self.trials = np.zeros(tuple([2 for _ in range(num_feats + 1)]))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
    
    def get_feats(self, state):
        features = []
        features.append(int(state['tree']['dist'] < 25))
        features.append(int(state['tree']['dist'] > 50))
        features.append(int(state['monkey']['bot'] > 330))
        features.append(int(state['monkey']['bot'] > 270))
        features.append(int(state['monkey']['bot'] < 60))
        features.append(int(state['monkey']['bot'] - state['tree']['bot'] < 20))
        features.append(int(state['tree']['top'] - state['monkey']['top'] < 0))
        features.append(int(state['monkey']['vel'] < 0))
        return tuple(features)
        
    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        action = int(npr.rand() < 0.5)
        feats = self.get_feats(state)
        if self.last_action != None:
            last_feats = self.get_feats(self.last_state)
            Qs = [self.Q[0][feats],self.Q[1][feats]]
            max_Q = np.max(Qs)
            new_epsilon = epsilon / max(self.trials[action][feats], 1)
            if (npr.rand() > new_epsilon):
                action = int(self.Q[1][feats] > self.Q[0][feats])
            self.Q[self.last_action][last_feats] += eta * (self.last_reward + discount_factor * max_Q - self.Q[self.last_action][last_feats])
        
        self.last_action = action
        self.last_state = state
        self.trials[feats] += 1
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
	run_games(agent, hist, 10000, 1)

	# Save history. 
	np.save('hist',np.array(hist))

