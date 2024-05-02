import random
import numpy as np

# the q learning intialization part
###
# Actions definition
ACTIONS = {
    'UP': 0,
    'DOWN': 1,
    'LEFT': 2,
    'RIGHT': 3,
    'NONE': 4
}
NUM_ACTIONS = len(ACTIONS)

# State dimensions
NUM_LVS = 5  # 1-5 levels for self and opponent
NUM_DIRECTIONS = 4  # front, left, back, right

# Assuming the distance has 3 possible states (0, 1, 2)
NUM_DISTANCE_STATES = 3

# State dimensions now including three closest foods and three closest garbages
STATE_DIMENSIONS = [
    NUM_LVS, NUM_LVS, NUM_DIRECTIONS, 
    NUM_DIRECTIONS, NUM_DIRECTIONS, NUM_DIRECTIONS,  # 1st, 2nd, 3rd closest food directions
    NUM_DIRECTIONS, NUM_DIRECTIONS, NUM_DIRECTIONS,  # 1st, 2nd, 3rd closest garbage directions
    NUM_DISTANCE_STATES  # Distance between player and opponent
]

# Initialize Q-table
INIT_VALUE = 10
# Q_table = np.full(STATE_DIMENSIONS + [NUM_ACTIONS], INIT_VALUE)

# leaning constants
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.9
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.999
###

FILE_PATH = "q_table3.npy"

def save_q_table(q_table, filename=FILE_PATH):
    np.save(filename, q_table)
    print(f"Q-table saved to {filename}")

def load_q_table(filename=FILE_PATH):
    try:
        q_table = np.load(filename)
        print(f"Loaded Q-table from {filename}")
        return q_table
    except FileNotFoundError:
        print("Q-table file not found, initializing a new one")
        return np.full(STATE_DIMENSIONS + [NUM_ACTIONS], INIT_VALUE)  # Using previously defined dimensions and initial values

def update_Q_table(Q_table, state_index, action_index, reward, new_state_index, alpha=0.1, gamma=0.9):
    current_q = Q_table[state_index][action_index]
    future_q = np.max(Q_table[new_state_index])
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * future_q)
    Q_table[state_index][action_index] = new_q

class MLPlay:
    def __init__(self,*args, **kwargs):
        print("Initial ml script")
        self.q_table = load_q_table()
        self.epsilon = EPSILON
        self.prev_score = 0  # Initialize previous score to 0
        self.prev_state_index = None  # To hold the previous state index
        self.prev_action_index = None  # To hold the previous action index

    def calculate_velocity(self, lv):
        # Velocity values based on level
        velocities = {1: 25, 2: 21, 3: 18, 4: 16, 5: 12, 6: 9}
        return velocities[lv]
    
    def calculate_average_velocity(self, self_lv, opponent_lv):
        self_vel = self.calculate_velocity(self_lv)
        opponent_vel = self.calculate_velocity(opponent_lv)
        return (self_vel + opponent_vel) / 2
    
    def quantize_distance(self, distance, avg_velocity):
        # Quantize distance based on average velocity and cap at 2
        quantized_distance = int(distance / avg_velocity)
        return min(quantized_distance, 2)  # Ensure the value doesn't exceed 2

    def calculate_direction(self, self_x, self_y, obj_x, obj_y):
        angle = np.degrees(np.arctan2(obj_y - self_y, obj_x - self_x)) % 360
        if 45 < angle <= 135:
            return 0  # front
        elif 135 < angle <= 225:
            return 1  # left
        elif 225 < angle <= 315:
            return 2  # back
        else:
            return 3  # right
        
    def get_state_index(self, scene_info):
        self_lv = scene_info['self_lv'] - 1  # Level 1-5 maps to indices 0-4
        opponent_lv = scene_info['opponent_lv'] - 1
        opponent_direction = self.calculate_direction(scene_info['self_x'], scene_info['self_y'], scene_info['opponent_x'], scene_info['opponent_y'])
        avg_velocity = self.calculate_average_velocity(scene_info['self_lv'], scene_info['opponent_lv'])
        distance_between = np.sqrt((scene_info['self_x'] - scene_info['opponent_x'])**2 + (scene_info['self_y'] - scene_info['opponent_y'])**2)
        distance_state = self.quantize_distance(distance_between, avg_velocity)

        # Initialize direction states
        food_directions = [None] * 3
        garbage_directions = [None] * 3

        items = sorted(scene_info['foods'], key=lambda x: np.sqrt((x['x'] - scene_info['self_x'])**2 + (x['y'] - scene_info['self_y'])**2))
        
        food_count = 0
        garbage_count = 0
        
        for item in items:
            if food_count < 3 and item['type'].startswith('FOOD'):
                food_directions[food_count] = self.calculate_direction(scene_info['self_x'], scene_info['self_y'], item['x'], item['y'])
                food_count += 1
            elif garbage_count < 3 and item['type'].startswith('GARBAGE'):
                garbage_directions[garbage_count] = self.calculate_direction(scene_info['self_x'], scene_info['self_y'], item['x'], item['y'])
                garbage_count += 1

            # Stop if all needed directions are filled
            if food_count == 3 and garbage_count == 3:
                break

        # Filling None values with a default, such as 'right' direction
        food_directions = [dir if dir is not None else 3 for dir in food_directions]
        garbage_directions = [dir if dir is not None else 3 for dir in garbage_directions]

        return tuple([self_lv, opponent_lv, opponent_direction] + food_directions + garbage_directions + [distance_state])


    def choose_action(self, state_index):
        if np.random.random() < self.epsilon:
            action = random.choice(list(ACTIONS.keys()))
        else:
            action = list(ACTIONS.keys())[np.argmax(self.q_table[state_index])]
        return action

    def update(self, scene_info: dict, *args, **kwargs): 
        """
        Update logic for Q-learning, considering deferred reward calculation.
        """
        current_score = scene_info['score']
        current_state_index = self.get_state_index(scene_info)
        action = self.choose_action(current_state_index)
        # print(scene_info['frame'], ':', action)
        action_index = ACTIONS[action]

        # print(f"Frame {scene_info['frame']}: Action - {action}, Epsilon - {self.epsilon}")

        if self.prev_state_index is not None and self.prev_action_index is not None:
            # Calculate reward based on score difference
            reward = current_score - self.prev_score
            # Update Q-table for the previous action taken from the previous state
            update_Q_table(self.q_table, self.prev_state_index, self.prev_action_index, reward, current_state_index, ALPHA, GAMMA)

        # Store the current state and score for the next update cycle
        self.prev_score = current_score
        self.prev_state_index = current_state_index
        self.prev_action_index = action_index

        # print("AI received data from game :", json.dumps(scene_info))
        # print(scene_info)

        # Epsilon decay
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

        return [action]

    def reset(self):
        """
        Reset the status
        """
        # Save the Q-table at the end of each episode
        save_q_table(self.q_table)
        self.prev_score = 0
        self.prev_state_index = None
        self.prev_action_index = None
        self.epsilon = EPSILON
        print("Resetting ML script")