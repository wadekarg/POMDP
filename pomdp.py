# Name: Wadekar, Gajanan
# Student ID: 100-162-9374
# Net ID: gxw9374
# Date: May 15, 2020

# importing all models
import numpy as np
import matplotlib.pyplot as plt
import random

# This class is the grid model of size 25x15 as required
class Grid:
    def __init__(self):
        # declaring te initial parameter for the state space model
        self.cols = 25
        self.rows = 15
        self.GAMMA = 0.95
        self.EPSILON = 0.1
        self.ALPHA = 0.5
        self.no_of_actions = 4  # Up, Down, Left, Right or N, S, E, W
        self.start_state = [0, 0]   # start state
        self.goal_states = [[14, 24]]    # goal state
        self.obstacles = [[0, 1], [2, 14], [2, 3], [4,14], [17, 8], [22, 2], [3, 3], [23, 1], [14,4], [24,0], [23,2], [6,7]]   # list of obstacles state
        self.max_runs = 100     # maximum episodes or runs untill we reach the goal
        self.beliefs = {}

    # give action at state, the function will return the next state
    def next_state(self, state, action):
        x, y = state
        reward = 0  # at start the reward is 0
        if action == 0:  # UP or N action
            x = max( x - 1, 0 )
        elif action == 1:  # DOWN or S action
            x = min( x + 1, self.rows - 1 )
        elif action == 2:  # LEFT or W action
            y = max( y - 1, 0 )
        elif action == 3:  # RIGHT or E action
            y = min( y + 1, self.cols - 1 )
        # if the next state is out of grid then assigning the reward to -100
        if x == state[0] and y == state[1]:
            reward = -100
        if [x, y] in self.obstacles:    # if state is in obstacles list, assigning the reward to -100
            x, y = state
            reward = -100
        if [x, y] in self.goal_states:  # if the next state is goal, assigning the reward to 100
            reward = 100.0
        return [x, y], reward

    def take_action(self, state, q_value_function):   # this function will take a action based on epsilon greedy method
        if random.random() < self.EPSILON:      # if random number is below epsilon then taking random action
            return np.random.randint( 0, 3 )
        else:
            max_values = np.max( q_value_function[state[0]][state[1]][:] )  # getting max values at q value for state
            list_of_max_values_indexes = [] # if the there are same max values then choosing random index from them
            for i in range( len( q_value_function[state[0]][state[1]][:] ) ):
                if q_value_function[state[0]][state[1]][i] == max_values:
                    list_of_max_values_indexes.append( i )
            return np.random.choice( list_of_max_values_indexes )

    # this function will decide if we should take the action taken or not based on transition probabilities
    def whether_to_take_action(self, action, orientation):
        p1 = random.random()  # for forward and backward
        p2 = random.random()  # for left and right
        # at the first I am assuming the orientation as UP
        if orientation in [0, 1]:  # so if for the first time it would be forward and down, backword
            if action in [0, 1]:  # when action is UP or DOWN
                if p1 <= 0.8:
                    return 1
                else:
                    return 0
            else:  # when action is LEFT or RIGHT i.e taking turn
                if p2 <= 0.9:
                    return 1
                else:
                    return 0
        else:  # if the orientation of agent is facing LEFT or RIGHT
            if action in [0, 1]:  # when action is UP or DOWN is like taking turn
                if p1 <= 0.9:
                    return 1
                else:
                    return 0
            else:  # when action is LEFT or RIGHT this means going forward or backward
                if p2 <= 0.8:
                    return 1
                else:
                    return 0

    # this function will compute the approximated function for action taken
    def function_approximation(self, action, q_values):
        sum_q_ba = 0
        key = list( self.beliefs.keys() )
        values = list( self.beliefs.values() )
        for r in range( len( key ) ):
            sum_q_ba += values[r] * q_values[key[r][0]][key[r][1]][action]
        return sum_q_ba
    # this function will get the observation probabilities
    def observation_probability(self, state, action):
        [x, y], rwd = self.next_state( state, action )
        if rwd == -100 or [x, y] == state:
            return 0
        else:
            return 1
    # this function will get the transition probability on taking action from one state to another
    def transition_probability(self, pre_state, state, action, pre_ori):
        # checking if action taken is forward/backward or right/left
        if action in [0, 1] and pre_ori in [0, 1]:      # if action is forward or backward
            if pre_state[0:2] == state:     # if agent is statying at same state even after taking action
                return 0.2
            else:
                return 0.8
        else:       # if action taken is turning left or right
            if pre_state[0:2] == state:     # if agent is statying at same state even after taking action
                return 0.1
            else:
                return 0.9

    # this function is for checking if the state is valid or not
    def if_valid_state(self, state):
        if state[0:2] in self.goal_states or state[0:2] in self.obstacles \
                or state[0] < 0 or state[0] >= self.rows or state[1] < 0 or state[1] >= self.cols:
            return 0
        else:
            return 1

    # computing the numerator of the formula of updating the beliefs
    def sigma_transition(self, next_state, action, orientation):
        sigma = 0
        list_previous_states = []  # this list will store the possible actions from where agent can reach the next state
        probable_list = []
        list_previous_states.append( [next_state[0], next_state[1] - 1, 3] )  # right
        list_previous_states.append( [next_state[0] + 1, next_state[1], 0] )  # up
        list_previous_states.append( [next_state[0], next_state[1] + 1, 2] )  # left
        list_previous_states.append( [next_state[0] - 1, next_state[1], 1] )  # down

        for i in range( 4 ):
            list_previous_states.append( [next_state[0], next_state[1], i] )
        for st in list_previous_states:
            if st[2] == action and self.if_valid_state( st ):
                tp = self.transition_probability( st, next_state, action, orientation )
                probable_list.append( st )
                if self.beliefs.get( (st[0], st[1]) ) is not None:
                    sigma += tp * self.beliefs.get( (st[0], st[1]) )
        return sigma, probable_list
    # updating the belief for probable states which could take us to next state
    def update_beliefs(self, state, next_state, action, orientation):
        p = self.observation_probability( state, action )
        sigma, prob_list_states = self.sigma_transition( next_state, action, orientation )
        deno = 0
        for each_state in prob_list_states:
            deno += sigma * self.observation_probability( [each_state[0], each_state[1]], each_state[2] )
        belief = {}
        if p != 0 and sigma != 0:
            belief[(next_state[0], next_state[1])] = p * sigma / deno
            self.beliefs = belief

if __name__ == '__main__':
    print( '*******************************Start***************************************' )
    grid = Grid()  # creating the grid object
    q_value_function = np.zeros( (grid.rows, grid.cols, grid.no_of_actions) )  # creating the q value function
    orientation = 0  # initial orientation of the agent
    run = 0  # variable to keep count of a run
    counting_steps = []  # storing the number of step taken in each run to reach the goal
    while (run < grid.max_runs):  # loop for number of run
        print('=======================================')
        run += 1
        grid.beliefs[(0, 0)] = 1  # setting belief at initial location as 0
        state = grid.start_state  # Setting a start state
        step = 0
        while state not in grid.goal_states:  # loop to count number of step taken in each run
            step += 1
            action = grid.take_action( [state[0], state[1], orientation], q_value_function )  # taking next action
            # if we can take action as per state space models transition probability requirement
            if grid.whether_to_take_action( action, orientation ) == 1:
                next_state, reward = grid.next_state( state, action )  # getting next state and reward as per action
                q_ba = grid.function_approximation( action, q_value_function )  # computing Q_ba for taken action
                b_s = grid.beliefs.get( (state[0], state[1]) )  # getting the belief at current state
                # updating the beliefs at next state or next probable states
                grid.update_beliefs( state, next_state, action, orientation )
                list_qc = []  # computing and storing Q values for action choices agent has at certain state
                for a in range( 4 ):
                    list_qc.append( grid.function_approximation( a, q_value_function ) )
                # computing q value and updating into the Q values table
                q_value_function[state[0], state[1], action] += (
                            grid.ALPHA * b_s * (reward + grid.GAMMA * max( list_qc ) - q_ba))
                # changing the orientation of the agent after taking next state
                orientation = action
                state = next_state
            else:
                continue
            if reward == 100:
                print( 'Episode: ', run, '    No. of steps taken: ', step )
                counting_steps.append( step )
                break
    # plotting the graph
    plt.plot( range( 0, grid.max_runs ), counting_steps, label='Graph of number of steps per episode' )
    plt.xlabel( 'run' )
    plt.ylabel( 'steps per episode' )
    plt.legend()
    plt.savefig( 'result.png' )
    plt.close()