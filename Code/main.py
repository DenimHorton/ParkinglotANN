from parkinglotgame import Parkinglot


# Easy representation of parkinglot
training_easy = Parkinglot(assigned_spots = [(3, 2)], ml_agent_type = "QLearningAgent")
print(f"Easy Parking Lot Training\n", "-----"*5)
training_easy.ml_agent.showParkinglot()
info_qlearning_easy = training_easy.ml_agent.parkCar(episodes = 1, verbose = False)
# print(training_easy.ml_agent.q_table)
print(info_qlearning_easy)
# Change agent methodolgy
print(f"Type of training agent: {type(training_easy.ml_agent)}")
training_easy.changeLearningAgentMethod("Deep Q-Learning")
print(f"Type of training agent: {type(training_easy.ml_agent)}")
info_deep_easy = training_easy.ml_agent.parkCar(episodes = 1, verbose = False)
# print(training_easy.ml_agent.q_table)
print(info_deep_easy)

# Hard representation of parkinglot
hard_barriers = [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8)]
training_hard = Parkinglot(pl_hight= 9,
                           pl_width=8,
                           brs=hard_barriers,
                           assigned_spots = [(3, 1)],
                           pl_spots={(3, 1):[-1, 0]}, 
                           ml_agent_type = "QLearningAgent")
print(f"Hard Parking Lot Training\n", "-----"*5)
training_hard.ml_agent.showParkinglot()
print(f"Type of training agent: {type(training_hard.ml_agent)}")
info_qlearning_hard = training_hard.ml_agent.parkCar(episodes = 1, verbose = False)
# print(training_hard.ml_agent.q_table)
print(info_qlearning_hard)
# Change agent methodolgy
training_hard.changeLearningAgentMethod("Deep Q-Learning")
print(f"Type of training agent: {type(training_hard.ml_agent)}")
info_deep_hard = training_hard.ml_agent.parkCar(episodes = 1, verbose = False)
# print(training_hard.ml_agent.q_table)
print(info_deep_hard)

