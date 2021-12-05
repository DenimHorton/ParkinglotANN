from parkinglotgame import Parkinglot

training = Parkinglot(ml_agent_type = "QLearningAgent")
# training.changeLearningAgentMethod("QLearningAgent")
# training.buildRewardTable()
# training.buildIllegalmovesTable()
print(type(training.ml_agent))
training.showParkinglot()

training.ml_agent.parkCar(episodes = 1, verbose = True)

print(training.ml_agent.q_table)

# training.performAction(training.pickRandomAction())
# training.performAction(training.pickGreedyAction())
