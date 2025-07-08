import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
#added
import random

STUDENT_ID = 'a1895261'
DEGREE = 'UG'  # or 'PG'


class MyAgent: #partially implemented
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen

        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

         #stores the agent experience (state, action, reward, next_state, done)                                      
        self.storage = [] # a data structure of your choice (D in the Algorithm 2)
        # A neural network MLP model which can be used as Q
        self.network = MLPRegression(input_dim=6, output_dim=2, learning_rate=1e-3)
        # network2 has identical structure to network1, network2 is the Q_f
        self.network2 = MLPRegression(input_dim=6, output_dim=2, learning_rate=1e-3)
        # initialise Q_f's parameter by Q's, here is an example
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 1.0 # probability ε in Algorithm 2 #full exploration
        self.epsilon_min = 0.05 #min value of epsilon
        self.epsilon_decay = 0.995 #how quickly exploration reduces
        self.n = 64 # the number of samples you'd want to draw from the storage each time
        self.discount_factor = 0.95 #gamma value for future reward discounting
        self.memory_limit = 10000 #max number of past experiences
        self.prev_state = None #the previous state
        self.prev_action= None #the previous action 
        self.output_dim = 2

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)
    

    def build_state(self, state):
        #question 2 in assignment description 
        #we are converting into feature vector
        #screen resolution for this flappy bird is 400*600 pixels
        if not state['pipes']:
            pipe_x = 400
            pipe_top = 0
            pipe_bottom=600
        else:
            pipe = state['pipes'][0]
            pipe_x = pipe['x']
            pipe_top = pipe['top']
            pipe_bottom = pipe['bottom']

        attributes = [
            state['bird_y'] / 600, #normalize bird vertical
            state['bird_velocity'] / 10, # 10 is the max normalization for volatility
            (pipe_x - state['bird_x']) / 400, #normalize horizontal distance to pipe
            pipe_top / 600, #normalize top of pipe
            pipe_bottom/ 600, #normalize bottom of pipe
            (pipe_bottom- pipe_top) / 600 #normalize height of pipe gap 
        ]
        return np.array(attributes, dtype=np.float32) 
    
    def reward(self, state):
        reward_a = 0.2 #motivation for bird to stay alive
        if state['done']:
            if state['done_type'] == 'hit_pipe': # reduce points for hitting pipe
                return -1.0
            elif state['done_type'] == 'offscreen': #reduce points for going off screen 
                return -2.0
            elif state['done_type'] == 'well_done':
                return 2.0
        return reward_a

    def encode_one_hot(self, index):
            #one hot encoding is a way to represent actions as a binary vector 
            #jump -> [1,0]
            #do_nothing -> [0,1]
            #this marks the chosen action and avoids continuous values 
            vector = np.zeros(self.output_dim)
            vector[index] = 1
            return vector
           
    def choose_action(self, state: dict, action_table: dict) -> int:
            """
            This function should be called when the agent action is requested.
            Args:
                state: input state representation (the state dictionary from the game environment)
                action_table: the action code dictionary
            Returns:
                action: the action code as specified by the action_table
            """
            # following pseudocode to implement this function
            state_representation = self.build_state(state).reshape(1,-1) #BUILD_STATE
            if self.mode == 'train' and np.random.uniform(0,1) < self.epsilon:
                a_t = random.randint(0,1) 
            else:
                quality =self.network.predict(state_representation) #predict q value from network 
                a_t = int(np.argmax(quality))#  max value action needs to be selected 
            self.prev_state = state_representation #now save the current state as previous
            self.prev_action = a_t
            return a_t
    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
            """
            This function should be called to notify the agent of the post-action observation.
            Args:
                state: post-action state representation (the state dictionary from the game environment)
                action_table: the action code dictionary
            Returns:
                None
            """
            next_state_representation = self.build_state(state).reshape(1,-1) #compute new state features

            reward = self.reward(state) #get the reward for the new state
            done = state['done']
            if self.mode == 'train':
                 self.storage.append((self.prev_state,self.prev_action, reward, next_state_representation)) #store the experience
                 if len(self.storage) > self.memory_limit: #make sure bugger doesnt exceed the limit
                      self.storage.pop(0)

                 if len(self.storage) > self.n:#if we have enough experience to train 
                    mbatch = random.sample(self.storage, self.n) #mini random batch 
                    inputs, targets, weights = [],[],[]
                
                    for previous_state ,action,reward, next_state in mbatch:
                        current_q = self.network.predict(previous_state.copy())[0]
                        next_q = self.network2.predict(next_state.copy())[0]
                        q_target = reward + self.discount_factor * np.max(next_q)

                        current_q[action] = q_target

                        inputs.append(previous_state[0])
                        targets.append(current_q)
                        weights.append(self.encode_one_hot(action))

                    self.network.fit_step(np.array(inputs), np.array(targets), np.array(weights))
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                    if random.random() < 0.05:
                        MyAgent.update_network_model(self.network2, self.network)

    
    def save_model(self, path: str = 'my_model.ckpt'):
            """
            Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

            Args:
                path: the full path to save the model weights, ending with the file name and extension

            Returns:

            """
            self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
            """
            Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
            Args:
                path: the full path to load the model weights, ending with the file name and extension

            Returns:

            """
            self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
            """
            Update one MLP model's model parameter by the parameter of another MLP model.
            Args:
                net_to_update: the MLP to be updated
                net_as_source: the MLP to supply the model parameters

            Returns:
                None
            """
            net_to_update.load_state_dict(net_as_source.state_dict())

    
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    args = parser.parse_args()

    # bare-bone code to train your agent (you may extend this part as well, we won't run your agent training code)
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level, game_length=10)
    agent = MyAgent(show_screen=False)
    best_score = 0
    episodes = 500
    for episode in range(episodes):
        env.play(player=agent)
            # env.score has the score value from the last play
            # env.mileage has the mileage value from the last play
        print(env.score)
        print(env.mileage)

            #check if the agent achieved the best reward so far

            # store the best model based on your judgement
        if env.score > best_score:
            best_score = env.score
            agent.save_model(path='my_model.ckpt')
             

            # you'd want to clear the memory after one or a few episode
            # you'd want to update the fixed Q-target network (Q_f) with Q's model parameter after one or a few episodes   
        # the below resembles how we evaluate your agent
        env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
        agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')
        scores =[]
        episodes = 10
        for episode in range(episodes):
            env2.play(player=agent2)
            scores.append(env2.score)

        print(np.max(scores))
        print(np.mean(scores))
