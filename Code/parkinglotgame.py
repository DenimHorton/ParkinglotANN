import numpy as np
import copy
from tabulate import tabulate
from learningagents import LearningAgent, DeepQLearningAgent, QLearningAgent


class Parkinglot:
    '''
    @param pl_height if the height of the parkinglot
    @param pl_width if the width of the parkinglot
    @param brs is the barries with in the parkinglot
    @param pl_spots are the parking spots in the parkinglot    
    '''
    def __init__(self, pl_hight= 7,  pl_width=6, brs=[(4,2), (4, 3), (4, 4), (4, 5)], assigned_spots = [(3, 2)], pl_spots={(3, 2):[-1, 0], (3, 3):[-1, 0], (3, 4):[-1, 0], (3, 5):[-1, 0], (5, 2):[1, 0], (5, 3):[1, 0], (5, 4):[1, 0], (5, 5):[1, 0]}, ml_agent_type = "LearningAgent"):
        # Define parkinglot deminsions. 
        self.parkinglot_height = pl_hight
        self.parkinglot_width = pl_width
        # Defualt parkinglot demensions provided from parkinglot model in README.md.
        # Define parkinglot barriers.
        self.barriers = brs
        # Define parkinglot spots set with exit vectors to enter and exit through.
        self.parking_spots = pl_spots
        # Define open/assigned parking spots.
        self.assigned_parking_spots = assigned_spots
        # Define parkinglot entrance where agent will begin from for each episode of training.
        self.parkinglot_entrance = (pl_hight, pl_width)
         # Define reward values as class attributes for easy manipulation.
        self.barrier_reward = -100
        self.border_reward = -100
        self.parking_spot_reward = -100
        self.parkinglot_reward = -1
        self.assigned_parking_reward = 100
        self.parkinglot_entrance_reward = -1
        # Intialize states for parkinglot.
        self.parkinglot = np.full((self.parkinglot_height+2, self.parkinglot_width+2), self.parkinglot_reward)
        # Set current locations at entrance
        self.current = self.parkinglot_entrance
        # Set barrier values.
        for barrier in self.barriers:
            self.parkinglot.itemset((barrier[0], barrier[1]), self.barrier_reward)
        # Set parking spot values.
        for parking_spot in self.parking_spots.keys():
            self.parkinglot.itemset(parking_spot, self.parking_spot_reward)
        # Set goal parkinglot spot.
        for open_parking_spot in self.assigned_parking_spots:
            self.parkinglot.itemset(open_parking_spot, self.assigned_parking_reward)
        # Set top & bottom border values.
        for boundry_state in range(len(self.parkinglot[0])):
            self.parkinglot.itemset((0, boundry_state), self.border_reward)
            self.parkinglot.itemset((self.parkinglot_height+1, boundry_state), self.border_reward)
        # Set right & left border values.
        for boundry_state in range(1, self.parkinglot_height+1):
            self.parkinglot.itemset((boundry_state, 0), self.border_reward)
            self.parkinglot.itemset((boundry_state, self.parkinglot_width+1), self.border_reward)
        # Set entrance state value
        self.parkinglot.itemset(self.parkinglot_entrance, self.parkinglot_entrance_reward)
        # List of actions along with mapped values ('DOWN', 'UP', 'RIGHT', or 'LEFT').  
        self.actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.actions_map = {(1, 0) : 'DOWN', (-1, 0): 'UP', (0, 1) : 'RIGHT', (0, -1): 'LEFT'}
        # Set illegal moves. 
        self.illegal_moves = {}   
        # Define learning agent.
        if ml_agent_type == "QLearningAgent":
            self.ml_agent = QLearningAgent(self)
        else:
            self.ml_agent = LearningAgent(self)
        # Define reward table values
        self.reward_table = {(x, y): list(self.parkinglot.copy().item((x, y)) for i in range(4)) for x in range(1, self.parkinglot_height+1) for y in range(1, self.parkinglot_width+1)}     
        # Define illegal moves.
        self.setIllegalMoves()
        
    def setIllegalMoves(self):
        for state in self.reward_table:
            for action in self.actions:
                rslt = np.add([state[0], state[1]], action)
                # Set illegal moves based on negative set rewards for BOUNDRIES & BARRIERS.
                if (self.parkinglot.item((rslt[0], rslt[1])) == self.border_reward) or (self.parkinglot.item((rslt[0], rslt[1])) == self.barrier_reward):
                    if state not in self.illegal_moves.keys():
                        self.illegal_moves[state] = [action]
                    else:
                        self.illegal_moves.get(state).append(action)
                # Set illegal moves for actions from NON-PARKSPOT to PARKSPOT that are not 
                # a specifically set vector in the parking_spots tupple.
                if (state not in self.parking_spots and (rslt[0], rslt[1]) in self.parking_spots):
                    inverse_entrnc_vec = list(np.array(self.parking_spots.get((rslt[0], rslt[1]))) * -1)
                    if (action != inverse_entrnc_vec):
                        if state not in self.illegal_moves.keys():
                            self.illegal_moves[state] = [inverse_entrnc_vec]
                        else:
                            self.illegal_moves.get(state).append(inverse_entrnc_vec)        
        # Set illegal moves for actions from PARKSPOT to PARKSPOT/NON-PARKSPOT.
        for spot in self.parking_spots:
            prk_entrnc = self.actions.copy()
            prk_entrnc.remove(self.parking_spots.get(spot))
            self.illegal_moves.update({spot:prk_entrnc})  

    def changeLearningAgentMethod(self, agent_type):
        if agent_type == "Q-Learning":
            self.ml_agent = QLearningAgent(self)
        elif agent_type == "Deep Q-Learning":
            self.ml_agent = DeepQLearningAgent(self)
        else:
            print("Not a valid learning agent.")
            
    # Resets the game to where agent is at entrance.
    def resetCar(self):
        self.current = self.parkinglot_entrance

    # def buildRewardTable(self):
    #     table = list([state, act_rwd_pair[0], act_rwd_pair[1], act_rwd_pair[2], act_rwd_pair[3]] for state, act_rwd_pair in self.reward_table.items())
    #     print(tabulate(table, headers=["States", 'DOWN', 'UP', 'RIGHT', 'LEFT' ], tablefmt="pretty"))

    # def buildIllegalmovesTable(self):
    #     table = list([state, bad_moves] for state, bad_moves in self.illegal_moves.items())
    #     for state in range(len(table)):
    #         for moves in range(1, len(table[state])):
    #             for move in range(len(table[state][moves])):
    #                 table[state][moves][move] = copy.copy(self.actions_map.get((table[state][moves][move][0], table[state][moves][move][1])))
    #     print(tabulate(table, headers=["States", 'Illegal Moves' ], tablefmt="pretty", stralign="left"))

    def showParkinglot(self):
        print(self.parkinglot)
        
    def step(self, action):
        # Previously occupied state. 
        prev_state = self.current
        # Matrix addition to get newly occupied state. .
        new_state_np = np.add(self.current, action)
        new_state = (new_state_np[0], new_state_np[1])
        # Reward for current occupied state in parkinglot environment.
        reward = self.parkinglot.item(new_state)
        # Check if car is parked in the right parking spot
        parked_car = self.carParked(new_state)
        return new_state, reward, parked_car, prev_state   
        
    def legalAction(self, action):
        # First check if stat even has illegal actions.
        if self.current not in self.illegal_moves.keys():
            return True
        elif action in self.illegal_moves.get(self.current):
            return False
        else:
            return True
    
    def carParked(self, state):
        if state in self.assigned_parking_spots:
            return True
        else:
            return False 
        
    def wrongParkingSpot(self, prk_spt):
        if prk_spt not in self.assigned_parking_spots:
            print(f"\t Wrong parking spot {prk_spt}!!!!")
            return True
        else:
            return False
        
        