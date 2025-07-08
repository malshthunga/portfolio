**Flappy Bird Reinforcement Learning**
![image](https://github.com/user-attachments/assets/66348b92-b118-4e8a-a100-2045cc58e317)

![image](https://github.com/user-attachments/assets/2fd4cf18-f89f-40c4-9ca1-3cc38038cda9)
**Installation**
Please run the following command in your terminal to install the needed Python libraries. Please note the Numpy version needs to be <2. If you already have installed the dependent Python libraries prior to this assignment, and you experience errors out of the box running the game, you should consider making a new virtualenv or a new Conda environment for this assignment as otherwise it could be a result of mismatching library versions. If you do not know how to do that, you may ask ChatGPT "How can I make a virtualenv/conda environment for a new Python project?".

pip install -r requirement.txt
Play the game
After installation, you can verify the game is functional by playing it yourself:

python3 play_game.py --level 1
The five levels of the game are:

Lv1 - "Sky is the limit."
There are "no pipes". As long as the bird does not drop out of the screen, the game continues. However, when you read the emulator fed state information, you will see it does contain pipes after the game starts for a period of time.
Lv2 - "Easy peasy lemon squeezy!"
The pipe openings (vertical gap between top/bottom pipes) stay level in the middle of the screen, the bird has a narrower space to navigate but it is still straightforward.
Lv3 - "Life has its ups and downs."
The pipe openings start to go up and down following a sine wave function through time. The agent needs to learn to jump strategically to accommodate the pipe opening changes.
Lv4 (UG) / Lv5 (PG) - "Life is full of random pipes."
The pipe openings are randomly positioned following a uniform distribution. This starts to resemble the original game.
Lv4 is for UG and Lv5 is for PG, PG's pipes are slightly wider to mildly increase the difficulty.
Lv6 - "Birdie thinks the pipes are getting mean!"
The pipes appear faster and the opening's vertical gap gets smaller.
The levels are defined in the config.yml together with other game parameters.

**Assignment Description**
Assignment objective
The assignment objective is for you to design an agent that can play the Flappy Bird game by itself. An agent class template has been created for you: my_agent.py. The game's increasing levels of difficulty are designed to gently guide you to master the craft of reinforcement learning.

Deep Q-learning Network (DQN) algorithm
In this assignment, we'd like you to implement the Deep Q-learning algorithm to train your agent. The original DQN algorithm is Algorithm 1 in the above link. It is a cool algorthm, if you are interested in what it is capable of, check this video Google DeepMind's Deep Q-learning playing Atari Breakout!. Implementing the full algorithm can be a bit difficult, therefore, we modified Algorithm 1 to Algorithm 2 below to reduce computational requirements and overall difficulty. If you are interested in what we modified, please check the content in the Helpful advice page. Reading the linked paper and lecture notes can help you understand the reinforcement learning concepts better. Another useful online resource to understand DQN is from HuggingFace.

![image](https://github.com/user-attachments/assets/c1c700ba-abe1-4f43-be76-03420650e268)


The blue colored lines are already implemented by us so your task is mainly implementing functions in the MyAgent class
located in my_agent.py. To quickly go through Algorithm 2, the main for-loop is defined below. You will see this section of code at the end of my_agent.py as well. As you may have noticed, Algorithm 2 is the agent training algorithm.
![image](https://github.com/user-attachments/assets/9d5b9685-9077-470c-b261-7c0ad971dff8)


The env.play(...) is a function of the emulator which handles the inner while-loop in Algorithm 2, please see console.py. You don't need to change any code in console.py, and we won't use your submitted console.py even if you upload it.
![image](https://github.com/user-attachments/assets/166f92de-0c38-4652-a99f-381f9a1618d5)

**Your task**
Please implement the choose_action(...) and after_action_observation(...) functions in my_agent.py to complete the modified DQN algorithm. The pseudocode of each is provided below.

![image](https://github.com/user-attachments/assets/6b43e13d-c845-420e-ba67-472b91fec44a)

In order to train your agent properly, you should also take care of the following bits.

Use your creativity and analytical skills to design the functions BUILD_STATE() and REWARD(). They are critical to your agent learning. We give you some advice over this at the beginning of the Helpful advice page.
Implementing the ONEHOT() function properly. Incorrect implementation of the function will directly affect the learning of the MLP.
The MyAgent.__init__(...) function is partially implemented for you. You should define the instance variables/attributes specified in the pseudocode of __init__(...) below.
You may add additional variables and helper functions as you need, but make sure they are written in the file my_agent.py.
You may want to clear the agent's memory D periodically, this is because we directly compute qt+1 before storing them in D. Those historical q values may become obsolete quickly as the network updates.
We assume some of you do not know how to train a deep learning neural network, if so you can use the MLP network defined in pytorch_mlp.py to implement the Q-value network. The MLP network is an imitation of MLPRegressor from scikit-learn. As MLPRegressor does not allow multiple regression outputs, we provide our own implementation to you to reduce the overall difficulty of the assignment. The MLPRegression has an example usage written at the end of the file. If you want to edit the MLP model, please feel free to do so, but please make sure the model file is not too large. The default network structure (200, 500, 100) is sufficiently large to handle the Flappy bird game and the saved model size is <1 MB.
![image](https://github.com/user-attachments/assets/b99622b1-c2f5-4aa0-bac9-2bbe281762c6)

