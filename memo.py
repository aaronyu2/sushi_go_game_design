from state import GameState, SushiCardType, card_to_index, score_dumplings, PUDDING_SCORE, NIGIRI_VALS, MAKI_SCORE
from lp import solve_lp, sample_from_mixed
from agent import Agent
import random
import copy
from mcts import eval_random_playout
import numpy as np
import scipy.optimize
import pickle

class MemoAgent(Agent):
    def __init__(self):
        self.memo = dict()

    def find_payoff_minmax(self, state, return_move=False):
        """Returns payoff for player 0 for a given state"""
        hash_key = state.hash_key()
        if hash_key in self.memo and not return_move:
            return self.memo[hash_key]
        if state.is_terminal():
            scores = state.score_played()
            return scores[0] - scores[1]
        p0_actions = state.legal_actions(0)
        p1_actions = state.legal_actions(1)
        
        # generate payoff matrix
        payoff_matrix = np.zeros((len(p0_actions), len(p1_actions)))
        for (p0_index, p0_action) in enumerate(p0_actions):
            for (p1_index, p1_action) in enumerate(p1_actions):
                moves = [p0_action, p1_action]
                cards_played = state.simulate_moves(moves)
                state_value = self.find_payoff_minmax(state)
                state.unsimulate_moves(moves, cards_played)
                payoff_matrix[p0_index][p1_index] = state_value

        # find p0 move        
        p0_payoff = None
        p0_best_action = None
        for p0_action in range(len(p0_actions)):
            min_payoff = None
            for p1_action in range(len(p1_actions)):
                if min_payoff is None or state_value < min_payoff:
                    min_payoff = state_value
            if p0_payoff is None or min_payoff > p0_payoff:
                p0_payoff = min_payoff
                p0_best_action = p0_action

        # find p1 move
        p1_payoff = None
        p1_best_action = None
        for p1_action in range(len(p1_actions)):
            min_payoff = None
            for p0_action in range(len(p0_actions)):
                if min_payoff is None or state_value > min_payoff:
                    min_payoff = state_value
            if p1_payoff is None or min_payoff < p1_payoff:
                p1_payoff = min_payoff
                p1_best_action = p1_action
        self.memo[hash_key] = payoff_matrix[p0_best_action][p1_best_action]

        # if len(state.hands[0]) >= 4:
        #     print(state.hands)
        #     print(state.played_cards)
        #     print(payoff_matrix)
        #     print("")
        
        if return_move:
            return payoff_matrix[p0_best_action][p1_best_action], p0_actions[p0_best_action], p1_actions[p1_best_action]
        return payoff_matrix[p0_best_action][p1_best_action]
    
    def find_payoff_lp(self, state, return_move=False):
        """Returns payoff for player 0 for a given state"""
        hash_key = state.hash_key()
        if hash_key in self.memo and not return_move:
            return self.memo[hash_key]
        if state.is_terminal():
            scores = state.score_played()
            return scores[0] - scores[1]
        p0_actions = state.legal_actions(0)
        p1_actions = state.legal_actions(1)
        
        # generate payoff matrix
        payoff_matrix = np.zeros((len(p0_actions), len(p1_actions)))
        for (p0_index, p0_action) in enumerate(p0_actions):
            for (p1_index, p1_action) in enumerate(p1_actions):
                moves = [p0_action, p1_action]
                cards_played = state.simulate_moves(moves)
                state_value = self.find_payoff_lp(state)
                state.unsimulate_moves(moves, cards_played)
                payoff_matrix[p0_index][p1_index] = state_value

        # find mixed strats
        p0_strat = np.asarray(solve_lp(payoff_matrix, p0_actions, p1_actions))
        p1_strat = np.asarray(solve_lp(-1 * np.transpose(payoff_matrix), p1_actions, p0_actions))
        payoff = p0_strat @ payoff_matrix @ p1_strat
        self.memo[hash_key] = payoff

        # if len(state.hands[0]) >= 4:
        #     print(state.hands)
        #     print(state.played_cards)
        #     print(payoff_matrix)
        #     print("")
        
        if return_move:
            p0_pure = sample_from_mixed(p0_strat)
            p1_pure = sample_from_mixed(p1_strat)
            return payoff, p0_actions[p0_pure], p1_actions[p1_pure]
        return payoff

    def play_turn(self, state, player_num):
        """Assumes 2 players"""
        payoff, p0_action, p1_action = self.find_payoff_lp(state, return_move=True)
        # print("-----------------------")
        if player_num == 0:
            return p0_action
        return p1_action
    