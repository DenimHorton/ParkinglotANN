from nerualnetwork import NeuralNetwork
import numpy as np 
import sys, time, random, copy
import matplotlib.pyplot as pypl   
from tqdm import tqdm

class LearningAgent(object):
    def __init__(self, parkinglot):
        # How well it values the reward received after a taken action
        self.learning_rate = 0.8
        # How we value future presents rather than presently taken action
        self.discount_factor = 0.95
        self.parkinglot = parkinglot
        self.epsilon = 0.5
        self.decay_factor = 0.999
        
    # Pick weighted 'random' action. 
    def pickRandomAction(self):
        choice = random.choice(self.parkinglot.actions)
        return choice
    
    # Pick 'greedy' action.
    def pickGreedyAction(self):
        # Refer to reward table and take best action based on rewards for all state:action paris
        actions = np.array(self.parkinglot.reward_table.get(self.parkinglot.current))
        max_action = float(np.max(actions))
        return self.parkinglot.actions[list(actions).index(max_action)]

    # Pefrom action.
    def performAction(self, action):
        result = np.add([self.parkinglot.current[0], self.parkinglot.current[1]], action)
        self.parkinglot.current = (result[0], result[1])   
    
    # Return true if explore step is to be taken with weight random action.
    def exploitOrExplore(self):
        if random.uniform(0, 1) > self.epsilon:
            return True
        else:
            return False
        
    def parkCar(self, episodes=10, verbose = False):
        info = {epi : list(information for information in range(3)) for epi in range(episodes)}
        # Perform set number of episodes (default set to 10).
        for episode in tqdm(range(1, episodes+1)):
            if verbose:
                print(f"\nEpisode {episode}\n-------------------------------------------------------")
            # Accumatlive reward reset for next episode.            
            total_rwrd = 0
            # Action count iterator.
            action_cnt = 0
            # Start timmer for episode.
            episode_tm_strt = time.time()
            # Continue making steps through the game specific to the ML method type.
            parked_card = False
            # og_epsilon = self.epsilon            
            # self.epsilon = og_epsilon 
            while not parked_card:
                # self.epsilon *= self.decay_factor
                legal_action = False
                # True means take exploring step 
                explore = self.exploitOrExplore()              
                if explore:
                    if verbose:
                        print("Exploration Action - {explore}")

                    # Continues selecting action untill legal move is found
                    while not legal_action:
                        action = self.pickRandomAction()
                        # Perform time step in parking car episode.
                        new_state, reward, parked_card, old_state = self.parkinglot.step(action)
                        # Update QTable when illegal move is made
                        self.updateRewardTable(old_state, reward, new_state, action, verbose=verbose)    
                        # Add onto accumaltive reward and actions taken.
                        total_rwrd += reward
                        action_cnt += 1                        
                        # Return True when its a legal action if not it returns false and we iterate through again untill a legal action is selected                            
                        legal_action = self.parkinglot.legalAction(action) 
                    if verbose:
                        print("Exploration Action - {explore}")
                    # Continues selecting action untill legal move is found
                    while not legal_action:
                        action = self.pickGreedyAction()
                        # Perform time step in parking car episode.
                        new_state, reward, parked_card, old_state = self.parkinglot.step(action)
                        # Update QTable when illegal move is made
                        self.updateRewardTable(old_state, reward, new_state, action, verbose=verbose)
                        # Return True when its a legal action if not it returns false and we iterate through again untill a legal action is selected
                        # Add onto accumaltive reward and actions taken.
                        total_rwrd += reward
                        action_cnt += 1                            
                        legal_action = self.parkinglot.legalAction(action) 

                # Move agent into new state with legal action from state          
                self.performAction(action)
                
                # Print current step state
                if verbose: 
                    print(f"\t step {action_cnt}:\t {self.parkinglot.current}")                
                
            # Stop timmer for episode & calculate time ellapsed.
            episode_tm_stp = time.time() 
            episode_tm_elpsd = episode_tm_stp - episode_tm_strt
            # Print information about episode
            if verbose:
                print("\t\t\t Car Parked!!!!")
                print("\t-----------------------------------------------")
                print(f"\tEpisode time:\t {episode_tm_elpsd}")
                print(f"\tSteps taken:\t {action_cnt}")
                print(f"\tTotal rewards:\t {total_rwrd}")
                print("\t----------------------------------------------")

            # Reset car for next episode
            self.parkinglot.resetCar()
            info.get(episode)[0] = episode_tm_elpsd
            info.get(episode)[1] = action_cnt
            info.get(episode)[2] = total_rwrd           
        return info
    
    def updateRewardTable(self, old_state, action_reward, new_state, action, verbose=False):
        
        # Find max reward from possbile actions to take.
        max_reward = np.max(np.array(self.parkinglot.reward_table.get(new_state)))
        # Refrence old value to update it to new return reward values for that state-action pair.
        old_value = self.parkinglot.reward_table.get(old_state)[self.parkinglot.actions.index(action)]
        
        # Verbose output for updates in Q-table.
        if verbose:
            print(f"\t\tPossible action rewards: {self.parkinglot.reward_table.get(new_state)}")
            print(f"\t\tAction: \t{self.parkinglot.actions[self.parkinglot.actions.index(action)]}")            
            print(f"\t\tPrevious state: {old_state}")
            print(f"\t\tNew state: \t{new_state}")
            print(f"\t\tReward: \t{action_reward}")
            print(f"\t\tUpdate \'{old_state} : {action}\' with parameters;")
            print(f"\t\t\tAction reward:\t\t{action_reward}")
            print(f"\t\t\tOriginal value:\t\t{old_value}")
            print(f"\t\t\tUpdating {old_state}:{action} = {old_value} + {action_reward} = {old_value + action_reward}")
            self.parkinglot.reward_table.get(old_state)[self.parkinglot.actions.index(action)] += action_reward            print(f"\t\tNew action values for {old_state} are now: {self.parkinglot.reward_table.get(old_state)}")    
        else:
            self.parkinglot.reward_table.get(old_state)[self.parkinglot.actions.index(action)] += action_reward

    def showParkinglot(self):    
        pypl.matshow(self.parkinglot.parkinglot)
        pypl.show(block=True)
        pypl.close()


class QLearningAgent(LearningAgent):
    def __init__(self, parkinglot):
        super(QLearningAgent,  self).__init__(parkinglot)
        # How well it values the reward received after a taken action
        self.learning_rate = 0.8
        # How we value future presents rather than presently taken action
        self.discount_factor = 0.95
        # Intilaize testing enviorment
        self.parkinglot = parkinglot
        self.epsilon = 0.5
        self.decay_factor = 0.999
        # Set Q-Table. 
        self.q_table = {(x, y): list(0.0 for i in range(4)) for x in range(self.parkinglot.parkinglot_height+2) for y in range(self.parkinglot.parkinglot_width+2)}
        # Define parameters for metrics and stuff like that.
        self.stats = f"{sys.path[0]}\\QLearning.txt"
        
    # Pefrom action.
    def performAction(self, action):
        result = np.add([self.parkinglot.current[0], self.parkinglot.current[1]], action)
        self.parkinglot.current = (result[0], result[1])           
        
    # Pick 'greedy' action.
    def pickGreedyAction(self):
        # Refer to Q-Table and take best action based on rewards for all state:action pairs.
        actions = np.array(self.q_table.get(self.parkinglot.current))
        max_action = float(np.max(actions))
        return self.parkinglot.actions[list(actions).index(max_action)]

    def updateQtable(self, old_state, action_reward, new_state, action, verbose = False):        
        # Find max reward from possbile actions to take.
        max_reward = np.max(np.array(self.q_table.get(new_state)))
        # Refrence old value to update it to new return reward values for that state-action pair.
        old_value = self.q_table.get(old_state)[self.parkinglot.actions.index(action)]
        
        # Verbose output for updates in Q-table.
        if verbose:
            print(f"\t\tPossible action rewards: {self.q_table.get(old_state)}")            
            print(f"\t\tAction: \t{self.parkinglot.actions[self.parkinglot.actions.index(action)]}")            
            print(f"\t\tPrevious state: {old_state}")
            print(f"\t\tNew state: \t{new_state}")
            print(f"\t\tReward: \t{action_reward}")
            print(f"\t\tUpdate \'{old_state} : {action}\' with parameters;")
            print(f"\t\t\tLearning rate:\t\t{self.learning_rate}")
            print(f"\t\t\tAction reward:\t\t{action_reward}")
            print(f"\t\t\tDiscount factor:\t{self.discount_factor}")
            print(f"\t\t\tMax reward (new state): {max_reward}")
            print(f"\t\t\tOriginal value:\t\t{old_value}")
            print(f"\t\t\tUpdating {old_state}:{action} = {old_value} + {self.learning_rate} * ({action_reward:.4f} + {self.discount_factor:.4f} * {max_reward:.4f} - {old_value:.4f}) = {old_value + self.learning_rate * (action_reward + self.discount_factor * max_reward - old_value)}")
            # Using the bellman equation to update Q-table.
            self.q_table.get(old_state)[self.parkinglot.actions.index(action)] += self.learning_rate * (action_reward + self.discount_factor * max_reward - old_value)
            print(f"\t\tNew action values for {old_state} are now: {self.q_table.get(old_state)}")
        else:
            # Using the bellman equation to update Q-table.
            self.q_table.get(old_state)[self.parkinglot.actions.index(action)] += self.learning_rate * (action_reward + self.discount_factor * max_reward - old_value)            

    def parkCar(self, episodes=10, verbose = False):
        info = []
        # Perform set number of episodes (default set to 10).
        for episode in tqdm(range(1, episodes+1)):
            info.clear()
            if verbose:
                print(f"\nEpisode {episode}\n-------------------------------------------------------")
            # Accumatlive reward reset for next episode.            
            total_rwrd = 0
            # Action count iterator.
            action_cnt = 1
            # Start timmer for episode.
            episode_tm_strt = time.time()
            # Continue making steps through the game specific to the ML method type.
            parked_card = False
            # # Value holder to reset at end of episodes if desired
            og_epsilon = self.epsilon            
            self.epsilon = og_epsilon 
            while not parked_card:
    
                legal_action = False
                # True means take exploring step 
                explore = self.exploitOrExplore()              
                if explore:                   
                    # Continues selecting action untill legal move is found
                    while not legal_action:
                        # Decaying epsilon which is our weighted values for how often to take exploited or exploritory actions      
                        self.epsilon *= self.decay_factor
                        # Print current step state
                        if verbose: 
                            print(f"\t Step {action_cnt}:\t CURRENT: {self.parkinglot.current}") 
                            print("Exploration Action")
                        action = self.pickRandomAction()
                        # Perform time step in parking car episode.
                        new_state, reward, parked_card, old_state = self.parkinglot.step(action)
                        # Update QTable when illegal move is made
                        self.updateQtable(old_state, reward, new_state, action, verbose=verbose)    
                        # Add onto accumaltive reward and actions taken.
                        total_rwrd += reward
                        action_cnt += 1                        
                        # Return True when its a legal action if not it returns false and we iterate through again untill a legal action is selected                            
                        legal_action = self.parkinglot.legalAction(action)
                else:
                    # Continues selecting action untill legal move is found
                    while not legal_action:
                        # Decaying epsilon which is our weighted values for how often to take exploited or exploritory actions      
                        self.epsilon *= self.decay_factor
                        # Print current step state
                        if verbose: 
                            print(f"\t Step {action_cnt}:\t CURRENT: {self.parkinglot.current}") 
                            print("Exploration Action")
                        action = self.pickGreedyAction()
                        # Perform time step in parking car episode.
                        new_state, reward, parked_card, old_state = self.parkinglot.step(action)
                        # Update QTable when illegal move is made
                        self.updateQtable(old_state, reward, new_state, action, verbose=verbose)
                        # Return True when its a legal action if not it returns false and we iterate through again untill a legal action is selected
                        # Add onto accumaltive reward and actions taken.
                        total_rwrd += reward
                        action_cnt += 1                            
                        legal_action = self.parkinglot.legalAction(action) 

                # Move agent into new state with legal action from state          
                self.performAction(action)         
                
            # Stop timmer for episode & calculate time ellapsed.
            episode_tm_stp = time.time() 
            episode_tm_elpsd = episode_tm_stp - episode_tm_strt
            # Print information about episode
            if verbose:
                print("\t\t\t Car Parked!!!!")
                print("\t-----------------------------------------------")
                print(f"\tEpisode time:\t {episode_tm_elpsd}")
                print(f"\tSteps taken:\t {action_cnt}")
                print(f"\tTotal rewards:\t {total_rwrd}")
                print("\t----------------------------------------------")

            # Reset car for next episode
            self.parkinglot.resetCar()
            info.append(episode)
            info.append(episode_tm_elpsd)
            info.append(action_cnt)
            info.append(total_rwrd)      
        return info
    
        
class DeepQLearningAgent(LearningAgent):
    def __init__(self, parkinglot):
        super(DeepQLearningAgent, self).__init__(parkinglot)
        # Set Neural Network as DEEP Q-LEARNING.
        self.ANN = NeuralNetwork(self.learning_rate)
        self.learning_rate = 0.11
        self.discount_factor = 0.95
        self.parkinglot = parkinglot
        self.epsilon = 0.95
        self.decay_factor = 0.99
        # Set Q-Table. 
        self.q_table = {(x, y): list(0.0 for i in range(4)) for x in range(self.parkinglot.parkinglot_height+2) for y in range(self.parkinglot.parkinglot_width+2)}
        # Define parameters for metrics and stuff like that.
        self.stats = f"{sys.path[0]}\\QLearning.txt"
        
    # Pefrom action.
    def performAction(self, action):
        result = np.add([self.parkinglot.current[0], self.parkinglot.current[1]], action)
        self.parkinglot.current = (result[0], result[1])           
        
    # Pick 'greedy' action.
    def pickGreedyAction(self):
        # Refer to Q-Table and take best action based on rewards for all state:action pairs.
        actions = np.array(self.q_table.get(self.parkinglot.current))
        max_action = float(np.max(actions))
        return self.parkinglot.actions[list(actions).index(max_action)]

    def updateQtable(self, old_state, action_reward, new_state, action, verbose = False):        
        # Find max reward from possbile actions to take.
        max_reward = np.max(np.array(self.q_table.get(new_state)))
        # Refrence old value to update it to new return reward values for that state-action pair.
        old_value = self.q_table.get(old_state)[self.parkinglot.actions.index(action)]
        
        # Verbose output for updates in Q-table.
        if verbose:
            print(f"\t\tPossible action rewards: {self.q_table.get(old_state)}")            
            print(f"\t\tAction: \t{self.parkinglot.actions[self.parkinglot.actions.index(action)]}")            
            print(f"\t\tPrevious state: {old_state}")
            print(f"\t\tNew state: \t{new_state}")
            print(f"\t\tReward: \t{action_reward}")
            print(f"\t\tUpdate \'{old_state} : {action}\' with parameters;")
            print(f"\t\t\tLearning rate:\t\t{self.learning_rate}")
            print(f"\t\t\tAction reward:\t\t{action_reward}")
            print(f"\t\t\tDiscount factor:\t{self.discount_factor}")
            print(f"\t\t\tMax reward (new state): {max_reward}")
            print(f"\t\t\tOriginal value:\t\t{old_value}")
            print(f"\t\t\tUpdating {old_state}:{action} = {old_value} + {self.learning_rate} * ({action_reward:.4f} + {self.discount_factor:.4f} * {max_reward:.4f} - {old_value:.4f}) = {old_value + self.learning_rate * (action_reward + self.discount_factor * max_reward - old_value)}")
            # Using the bellman equation to update Q-table.
            self.q_table.get(old_state)[self.parkinglot.actions.index(action)] += self.learning_rate * (action_reward + self.discount_factor * max_reward - old_value)
            print(f"\t\tNew action values for {old_state} are now: {self.q_table.get(old_state)}")
        else:
            # Using the bellman equation to update Q-table.
            self.q_table.get(old_state)[self.parkinglot.actions.index(action)] += self.learning_rate * (action_reward + self.discount_factor * max_reward - old_value)            

    def parkCar(self, episodes=10, verbose = False):
        info = []
        # Perform set number of episodes (default set to 10).
        for episode in tqdm(range(1, episodes+1)):
            info.clear()
            if verbose:
                print(f"\nEpisode {episode}\n-------------------------------------------------------")
            # Accumatlive reward reset for next episode.            
            total_rwrd = 0
            # Action count iterator.
            action_cnt = 1
            # Start timmer for episode.
            episode_tm_strt = time.time()
            # Continue making steps through the game specific to the ML method type.
            parked_card = False
            # # Value holder to reset at end of episodes if desired
            og_epsilon = self.epsilon            
            self.epsilon = og_epsilon 
            while not parked_card:
    
                legal_action = False
                # True means take exploring step 
                explore = self.exploitOrExplore()              
                if explore:                   
                    # Continues selecting action untill legal move is found
                    while not legal_action:
                        # Decaying epsilon which is our weighted values for how often to take exploited or exploritory actions      
                        self.epsilon *= self.decay_factor
                        # Print current step state
                        if verbose: 
                            print(f"\t Step {action_cnt}:\t CURRENT: {self.parkinglot.current}") 
                            print("Exploration Action")
                        action = self.pickRandomAction()
                        # Perform time step in parking car episode.
                        new_state, reward, parked_card, old_state = self.parkinglot.step(action)
                        # Update QTable when illegal move is made
                        self.updateQtable(old_state, reward, new_state, action, verbose=verbose)    
                        # Add onto accumaltive reward and actions taken.
                        total_rwrd += reward
                        action_cnt += 1                        
                        # Return True when its a legal action if not it returns false and we iterate through again untill a legal action is selected                            
                        legal_action = self.parkinglot.legalAction(action)
                else:
                    # Continues selecting action untill legal move is found
                    while not legal_action:
                        # Decaying epsilon which is our weighted values for how often to take exploited or exploritory actions      
                        self.epsilon *= self.decay_factor
                        # Print current step state
                        if verbose: 
                            print(f"\t Step {action_cnt}:\t CURRENT: {self.parkinglot.current}") 
                            print("Exploration Action")
                        action = self.pickGreedyAction()
                        # Perform time step in parking car episode.
                        new_state, reward, parked_card, old_state = self.parkinglot.step(action)
                        # Update QTable when illegal move is made
                        self.updateQtable(old_state, reward, new_state, action, verbose=verbose)
                        # Return True when its a legal action if not it returns false and we iterate through again untill a legal action is selected
                        # Add onto accumaltive reward and actions taken.
                        total_rwrd += reward
                        action_cnt += 1                            
                        legal_action = self.parkinglot.legalAction(action) 

                # Move agent into new state with legal action from state          
                self.performAction(action)         
                
            # Stop timmer for episode & calculate time ellapsed.
            episode_tm_stp = time.time() 
            episode_tm_elpsd = episode_tm_stp - episode_tm_strt
            # Print information about episode
            if verbose:
                print("\t\t\t Car Parked!!!!")
                print("\t-----------------------------------------------")
                print(f"\tEpisode time:\t {episode_tm_elpsd}")
                print(f"\tSteps taken:\t {action_cnt}")
                print(f"\tTotal rewards:\t {total_rwrd}")
                print("\t----------------------------------------------")

            # Reset car for next episode
            self.parkinglot.resetCar()
            info.append(episode)
            info.append(episode_tm_elpsd)
            info.append(action_cnt)
            info.appen        

        
    def convertToNeuralInput(self):
        pass
