from parkinglotgame import Parkinglot

class LearningAgent:
    def __init__(self):
        # Assign barrier & border reward values
        self.barrier_reward = -10
        self.border_reward = -10
        self.parking_spot_reward = 0
        self.parkinglot_reward = -1.0
        self.assigned_parking_reward = 10