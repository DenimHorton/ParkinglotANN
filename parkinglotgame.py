import numpy as np
import random, time, copy, sys
from tabulate import tabulate
from os import system, name
import matplotlib.pyplot as pypl 


class Parkinglot:
  def __init__(self):
    # Set parkinglot deminsions 
    self.parkinglot_height = 7
    self.parkinglot_width = 6
    
    # Defualt parking lot provided in README.txt
    # Intialize states for parking lot with buffer barriers rows and columns
    self.parkinglot = np.full((self.parkinglot_height+2, self.parkinglot_width+2), self.parkinglot_reward)
    # Set parking lot barriers
    self.barriers = [(4,2), (4, 3), (4, 4), (4, 5)]
    # Set parking lot spots set with exit vectors to enter and exit through
    self.parking_spots = {(3, 2):[-1, 0], (3, 3):[-1, 0], (3, 4):[-1, 0], (3, 5):[-1, 0], (5, 2):[1, 0], (5, 3):[1, 0], (5, 4):[1, 0], (5, 5):[1, 0]}
    # Set barrier values
    for barrier in self.barriers:
      self.parkinglot.itemset((barrier[0], barrier[1]), self.barrier_reward)
    # Set parking spot values
    for parking_spot in self.parking_spots.keys():
      self.parkinglot.itemset(parking_spot, self.parking_spot_reward)
    # Set goal parkinglot spot
    self.assigned_parking_spot = (3, 2) 
    self.parkinglot.itemset(self.assigned_parking_spot, self.assigned_parking_reward)
    # Set top & bottom border values
    for boundry_state in range(len(self.parkinglot[0])):
      self.parkinglot.itemset((0, boundry_state), self.border_reward)
      self.parkinglot.itemset((self.parkinglot_height+1, boundry_state), self.border_reward)
    # Set right & left border values
    for boundry_state in range(1, self.parkinglot_height+1):
      self.parkinglot.itemset((boundry_state, 0), self.border_reward)
      self.parkinglot.itemset((boundry_state, self.parkinglot_width+1), self.border_reward)
    # Set entrance state
    self.parkinglot_entrance = (7, 6)
    self.parkinglot.itemset(self.parkinglot_entrance, 0)
    # Set parkinglot entrance
    self.current = self.parkinglot_entrance
  
    # Set actions ('DOWN', 'UP', 'RIGHT', or 'LEFT')  
    self.actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    self.actions_map = {(1, 0) : 'DOWN', (-1, 0): 'UP', (0, 1) : 'RIGHT', (0, -1): 'LEFT'}
    # Set Q-Table 
    self.q_table = {(x, y): list(0.0 for i in range(5)) for x in range(1, self.parkinglot_height+1) for y in range(1, self.parkinglot_width+1)}
    # Set illegal moves 
    self.illegal_moves = {}

  # Resets the game to where agent is at entrance
  def resetGame(self):
    self.current = self.parkinglot_entrance

  # Play game
  def playGame(self, agent_type, learning_rate = 0.05, discount_factor = 0.90, epsilon = 0.5 episodes = 100):
    pass

  # Pick a random action to perfom from current state 
  def pickRandomAction(self, wighted_radn):
    choice = random.choices(population=self.actions, k=wighted_radn)    
    return choice

  def pickGreedyAction(self):
    # Build temp array of neghboring states
    choice = np.max(self.q_table.get((self.current[0], self.current[1])))
    return choice

  def performAction(self, action):
    self.current = np.add(self.current, action)

  def updateQtable(self):
    pass

  def buildQTable(self):
    table = list([state, act_rwd_pair[0], act_rwd_pair[1], act_rwd_pair[2], act_rwd_pair[3]] for state, act_rwd_pair in self.q_table.items())
    print(tabulate(table, headers=["States", 'DOWN', 'UP', 'RIGHT', 'LEFT' ], tablefmt="pretty"))

  def buildIllegalmovesTable(self):
    table = list([state, bad_moves] for state, bad_moves in self.illegal_moves.items())
    # print(table)
    for state in range(len(table)):
      for moves in range(1, len(table[state])):
        for move in range(len(table[state][moves])):
          table[state][moves][move] = self.actions_map.get((table[state][moves][move][0], table[state][moves][move][1]))
    print(tabulate(table, headers=["States", 'Illegal Moves' ], tablefmt="pretty", stralign="left"))

  def trainingGUI(self):
    try:
      while True:
        print("idk")
    except KeyboardInterrupt:
      sys.exit(-1)

  def showParkinglot(self, pl):
    pypl.matshow(pl)
    pypl.show(block=False)
    time.sleep(1)
    pypl.close()
    print(self.parkinglot, end='')


  def setIllegalMoves(self):
    for state in self.q_table:
      for action in self.actions:
        rslt = np.add([state[0], state[1]], action)

        # Set illegal moves based on negative set rewards for BOUNDRIES & BARRIERS
        if (self.parkinglot.item((rslt[0], rslt[1])) == self.border_reward) or (self.parkinglot.item((rslt[0], rslt[1])) == self.barrier_reward):
          if state not in self.illegal_moves.keys():
            self.illegal_moves[state] = [action]
          else:
            self.illegal_moves.get(state).append(action)

        # Set illegal moves for actions from NON-PARKSPOT to PARKSPOT that are not 
        # a specifically set vector in the parking_spots tupple
        if (state not in self.parking_spots and (rslt[0], rslt[1]) in self.parking_spots):
          inverse_entrnc_vec = list(np.array(self.parking_spots.get((rslt[0], rslt[1]))) * -1)
          if (action != inverse_entrnc_vec):
            if state not in self.illegal_moves.keys():
              self.illegal_moves[state] = [inverse_entrnc_vec]
            else:
              self.illegal_moves.get(state).append(inverse_entrnc_vec)


    # Set illegal moves for actions from PARKSPOT to PARKSPOT/NON-PARKSPOT
    for spot in self.parking_spots:
      prk_entrnc = self.actions.copy()
      prk_entrnc.remove(self.parking_spots.get(spot))
      self.illegal_moves.update({spot:prk_entrnc})       

game = Parkinglot()
game.setIllegalMoves()
game.buildQTable()
game.buildIllegalmovesTable()
game.showParkinglot(game.parkinglot)

# game.performAction(game.pickRandomAction())
game.performAction(game.pickGreedyAction())
