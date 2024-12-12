from state import GameState, SushiCardType, score_round, score_pudding
from agent import Agent
import random
import copy
from mcts import eval_random_playout
import numpy as np
import scipy.optimize
import pickle

def solve_lp(start_matrix, p0_actions, p1_actions, collector=None, state=None):
    """
    Finds best mixed strategy for p0.
    Args:
        start_matrix: np 2d array of shape (len(p0_actions), len(p1_actions))
    """
    # define linear program
    payoff_matrix = np.pad(np.transpose(start_matrix), ((0, 2), (0, 1)), mode='constant', constant_values=0)
    payoff_matrix *= -1
    payoff_matrix[len(p1_actions), :] = 1
    payoff_matrix[len(p1_actions) + 1, :] = -1
    payoff_matrix[:, len(p0_actions)] = 1
    payoff_matrix[len(p1_actions):len(p1_actions) + 2, len(p0_actions)] = 0
    b_ub = np.asarray([0.0] * len(p1_actions) + [1.0, -1.0])
    c = [0.0] * len(p0_actions) + [-1.0]
    bounds = [(0, 1)] * len(p0_actions) + [(None, None)]

    # solve linear program
    result = scipy.optimize.linprog(c, payoff_matrix, b_ub, None, None, bounds)
    # value = 1.0 / result.fun if result.fun != 0 else 1
    # x = [xi * value for xi in result.x]
    # strat = np.argmin(result.x[:-1])
    
    # save data for logging
    if collector is not None and state is not None:
        num_pure = 0
        for i in range(len(result.x) - 1):
            if not np.isclose(result.x[i], 0.0):
                num_pure += 1
        collector.save_lp_pure(num_pure, state)
    return result.x[:-1]

def sample_from_mixed(mixed_strat):
    rand_val = np.random.rand(1)[0]
    total = 0
    for i in range(len(mixed_strat)):
        total += mixed_strat[i]
        if total >= rand_val:
            return i
    raise ValueError("Not a valid probability distribution")


class LPAgent(Agent):
    def __init__(self, num_playouts, collector):
        self.num_playouts = num_playouts
        self.collector = collector

    def play_turn(self, state, player_num):
        """Assumes 2 players"""
        # populate payoff matrix
        my_actions = state.legal_actions(player_num)
        their_actions = state.legal_actions(1 - player_num)
        payoff_matrix = np.zeros((len(my_actions), len(their_actions)))
        for (my_index, my_action) in enumerate(my_actions):
            for (their_index, their_action) in enumerate(their_actions):
                if player_num == 0:
                    moves = [my_action, their_action]
                else:
                    moves = [their_action, my_action]
                cards_played = state.simulate_moves(moves)
                state_value = eval_random_playout(state, player_num, self.num_playouts)
                state.unsimulate_moves(moves, cards_played)
                payoff_matrix[my_index][their_index] = state_value
        
        # solve linear program
        mixed_strat = solve_lp(payoff_matrix, my_actions, their_actions, collector=self.collector, state=state)

        # sample from mixed strategy
        strat = sample_from_mixed(mixed_strat)
        
        # print(result.status, result.fun)
        # print(result.x, result.fun, strat)
        # print(state.hands)
        # print(payoff_matrix)
        return my_actions[strat]
    

    
        

