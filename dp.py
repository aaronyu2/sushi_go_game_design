from state import GameState, SushiCardType, card_to_index, score_dumplings, PUDDING_SCORE, NIGIRI_VALS, MAKI_SCORE
from lp import solve_lp, sample_from_mixed
from agent import Agent
import random
import copy
from mcts import eval_random_playout
import numpy as np
import scipy.optimize
import pickle

index_to_card = dict()
for (card, index) in card_to_index.items():
    index_to_card[index] = card
NUM_UNIQUE_CARDS = len(card_to_index)
CHOPSTICKS_INDEX = card_to_index[SushiCardType.CHOPSTICKS]
WASABI_INDEX = card_to_index[SushiCardType.WASABI]
PUDDING_HASH_INDEX = [NUM_UNIQUE_CARDS * 2 + card_to_index[SushiCardType.PUDDING], NUM_UNIQUE_CARDS * 3 + card_to_index[SushiCardType.PUDDING]]
nigiri_to_wasabi = {
    card_to_index[SushiCardType.EGG_NIGIRI]: card_to_index[SushiCardType.EGG_WASABI],
    card_to_index[SushiCardType.SALMON_NIGIRI]: card_to_index[SushiCardType.SALMON_WASABI],
    card_to_index[SushiCardType.SQUID_NIGIRI]: card_to_index[SushiCardType.SQUID_WASABI],
}
wasabi_to_nigiri = dict()
for (nigiri, wasabi) in nigiri_to_wasabi.items():
    wasabi_to_nigiri[wasabi] = nigiri

def expand_hash_state(hash_state):
    hands = [[hash_state[i + (NUM_UNIQUE_CARDS * player_num)] for i in range(NUM_UNIQUE_CARDS)] for player_num in range(2)]
    played = [[hash_state[i + (NUM_UNIQUE_CARDS * (2 + player_num))] for i in range(NUM_UNIQUE_CARDS)] for player_num in range(2)]
    return hands, played

def hash_state_legal_actions(hash_state):
    """
    Returns a list of all legal moves for both players for a given hash state.
    Currently assumes 2 players.
    Assumes chopsticks are first in the card order of hashed states.
    """
    hands, played = expand_hash_state(hash_state)
    # Find legal moves for both players
    legal_moves = [[] for i in range(2)]
    for player_num in range(2):
        for i in range(NUM_UNIQUE_CARDS):
            if hands[player_num][i] > 0:
                legal_moves[player_num].append([i])
        if played[player_num][CHOPSTICKS_INDEX] > 0:
            for i in range(1, NUM_UNIQUE_CARDS):
                if hands[player_num][i] > 1:
                    legal_moves[player_num].append([i, i])
            for i in range(1, NUM_UNIQUE_CARDS):
                for j in range(i + 1, NUM_UNIQUE_CARDS):
                    if hands[player_num][i] > 0 and hands[player_num][j] > 0:
                        legal_moves[player_num].append([i, j])
    return legal_moves

def simulate_moves(hands, played, p0_move, p1_move):
    moves_made = [[card for card in move] for move in (p0_move, p1_move)]
    for player_num in range(2):
        if player_num == 0:
            move = p0_move
        else:
            move = p1_move
        for i, card in enumerate(move):
            hands[player_num][card] -= 1
            if played[player_num][WASABI_INDEX] > 0 and (card == card_to_index[SushiCardType.EGG_NIGIRI] or card == card_to_index[SushiCardType.SALMON_NIGIRI] or card == card_to_index[SushiCardType.SQUID_NIGIRI]):
                played[player_num][nigiri_to_wasabi[card]] += 1
                played[player_num][WASABI_INDEX] -= 1
                moves_made[player_num][i] = nigiri_to_wasabi[card]
            else:
                played[player_num][card] += 1
        if len(move) == 2:
            played[player_num][CHOPSTICKS_INDEX] -= 1
            hands[player_num][CHOPSTICKS_INDEX] += 1
    return moves_made

def unsimulate_moves(hands, played, moves_made):
    for player_num in range(2):
        move = moves_made[player_num]
        for card in move:
            if card == card_to_index[SushiCardType.EGG_WASABI] or card == card_to_index[SushiCardType.SALMON_WASABI] or card == card_to_index[SushiCardType.SQUID_WASABI]:
                played[player_num][card] -= 1
                played[player_num][WASABI_INDEX] += 1
                hands[player_num][wasabi_to_nigiri[card]] += 1
            else:
                hands[player_num][card] += 1
                played[player_num][card] -= 1
        if len(move) == 2:
            played[player_num][CHOPSTICKS_INDEX] += 1
            hands[player_num][CHOPSTICKS_INDEX] -= 1

def hash_state_succs(hash_state):
    """
    Returns a list of all successors to a given hash state.
    Currently assumes 2 players.
    Assumes chopsticks are first in the card order of hashed states.
    """
    hands, played = expand_hash_state(hash_state)
    # Find legal moves for both players
    legal_moves = hash_state_legal_actions(hash_state)
    
    # Iterate over all successors
    succs = set()
    for p0_move in legal_moves[0]:
        for p1_move in legal_moves[1]:
            # make moves
            moves_made = simulate_moves(hands, played, p0_move, p1_move)
            # add to result
            succs.add(tuple(hands[1] + hands[0] + played[0] + played[1]))
            # unmake moves
            unsimulate_moves(hands, played, moves_made)
    return succs

def score_hash_state(hash_state, score_pudding=True):
    """Returns p0 score - p1 score given a hash state. Assumes 2 players."""
    score_diff = 0
    played = [[hash_state[i + (NUM_UNIQUE_CARDS * (2 + player_num))] for i in range(NUM_UNIQUE_CARDS)] for player_num in range(2)]
    # score pudding
    if score_pudding:
        pudding_diff = played[0][card_to_index[SushiCardType.PUDDING]] - played[1][card_to_index[SushiCardType.PUDDING]]
        if pudding_diff > 0:
            score_diff += PUDDING_SCORE
        elif pudding_diff < 0:
            score_diff -= PUDDING_SCORE
    # score tempura
    score_diff += 5 * ((played[0][card_to_index[SushiCardType.TEMPURA]] // 2) - (played[1][card_to_index[SushiCardType.TEMPURA]] // 2))
    # score sashimi
    score_diff += 10 * ((played[0][card_to_index[SushiCardType.SASHIMI]] // 3) - (played[1][card_to_index[SushiCardType.SASHIMI]] // 3))
    # score dumplings
    score_diff += score_dumplings(played[0][card_to_index[SushiCardType.DUMPLING]]) - score_dumplings(played[1][card_to_index[SushiCardType.DUMPLING]])
    # score maki
    maki_0 = 3 * played[0][card_to_index[SushiCardType.MAKI_ROLLS_3]] + 2 * played[0][card_to_index[SushiCardType.MAKI_ROLLS_2]] + played[0][card_to_index[SushiCardType.MAKI_ROLLS_1]]
    maki_1 = 3 * played[1][card_to_index[SushiCardType.MAKI_ROLLS_3]] + 2 * played[1][card_to_index[SushiCardType.MAKI_ROLLS_2]] + played[1][card_to_index[SushiCardType.MAKI_ROLLS_1]]
    # both score 3 or 0 if number is equal, but score diff is unchanged when that happens
    if maki_0 > maki_1:
        score_diff += MAKI_SCORE[0] - MAKI_SCORE[1]
        if maki_1 == 0:
            score_diff += MAKI_SCORE[1]
    elif maki_1 > maki_0:
        score_diff -= MAKI_SCORE[0] - MAKI_SCORE[1]
        if maki_0 == 0:
            score_diff -= MAKI_SCORE[1]
    # score nigiri
    for card in NIGIRI_VALS:
        score_diff += NIGIRI_VALS[card] * (played[0][card_to_index[card]] - played[1][card_to_index[card]])
    
    return score_diff

def hash_action_to_index_action(action, hand):
    if len(action) == 1:
        card_type = index_to_card[action[0]]
        return [hand.index(card_type)]
    if action[0] == action[1]:
        ans = []
        card_type = index_to_card[action[0]]
        for i in range(len(hand)):
            if hand[i] == card_type:
                ans.append(i)
                if len(ans) == 2:
                    return ans
        return ans
    else:
        # if playing wasabi with nigiri, return the wasabi first
        if action[1] == WASABI_INDEX and action[0] in nigiri_to_wasabi:
            return [hand.index(index_to_card[action[1]]), hand.index(index_to_card[action[0]])]
        return [hand.index(index_to_card[action[0]]), hand.index(index_to_card[action[1]])]
    
def hash_add_puddings(hash_state, p0_puddings, p1_puddings):
    return (hash_state[PUDDING_HASH_INDEX[0]] 
        + tuple(hash_state[PUDDING_HASH_INDEX[0]] + p0_puddings) 
        + hash_state[PUDDING_HASH_INDEX[0] + 1:PUDDING_HASH_INDEX[1]] 
        + tuple(hash_state[PUDDING_HASH_INDEX[1]] + p1_puddings) 
        + hash_state[PUDDING_HASH_INDEX[1] + 1:])

class DPAgent(Agent):
    def __init__(self, collector=None, num_rounds=1, is_score=True):
        self.dp = [[] for i in range(num_rounds)]
        self.collector = collector
        self.num_rounds = num_rounds
        self.curr_state = None
        self.is_score = is_score
        self.round_num = 0
        self.turn_num = 0
        # keyed by round_num, then by number of puddings player 0 has going into the round
        self.starting_payoffs = [dict() for i in range(num_rounds)]

    def find_all_states(self, start_states, round_num):
        """Finds all possible states reachable from starting_state and enters them as keys into self.dp"""
        self.dp[round_num].append(dict())
        for hash_state in start_states:
            self.dp[round_num][-1][hash_state] = None
        
        for turn_num in range(self.cards_per_round):
            curr_states = dict()
            for prev_state in self.dp[round_num][-1]:
                # look at all previous states and add all their possible successors to curr_states as keys
                succs = hash_state_succs(prev_state)
                for succ in succs:
                    curr_states[succ] = None
            self.dp[round_num].append(curr_states)
    
    def fill_dp(self, round_num):
        for hash_state in self.dp[round_num][-1]:
            if round_num == self.num_rounds - 1:
                # find payoff for terminal states (no rounds after)
                if self.is_score:
                    self.dp[round_num][-1][hash_state] = score_hash_state(hash_state)
                else:
                    score_diff = score_hash_state(hash_state)
                    if score_diff > 0:
                        self.dp[round_num][-1][hash_state] = 1
                    elif score_diff < 0:
                        self.dp[round_num][-1][hash_state] = -1
                    else:
                        self.dp[round_num][-1][hash_state] = 0
            else:
                # find payoff for ending states of earlier rounds
                self.dp[round_num][-1][hash_state] = self.starting_payoffs[round_num + 1][hash_state[PUDDING_HASH_INDEX[0]]] + score_hash_state(hash_state, score_pudding=False)
        # find payoff for nonterminal states
        for dp_index in range(len(self.dp[round_num]) - 2, -1, -1):
            for hash_state in self.dp[round_num][dp_index]:
                self.dp[round_num][dp_index][hash_state] = self.find_payoff_lp(hash_state, dp_index, round_num)
                if dp_index == 0:
                    self.starting_payoffs[round_num][hash_state[PUDDING_HASH_INDEX[0]]] = self.dp[round_num][dp_index][hash_state]

    
    def find_payoff_lp(self, hash_state, dp_index, round_num, return_move_player=None):
        """Returns payoff for player 0 for a given state"""
        hands, played = expand_hash_state(hash_state)
        legal_moves = hash_state_legal_actions(hash_state)
        p0_actions = legal_moves[0]
        p1_actions = legal_moves[1]
        # next_dp_index = len(self.dp[round_num]) - sum(hash_state[:NUM_UNIQUE_CARDS])
        next_dp_index = dp_index + 1
        
        # generate payoff matrix
        payoff_matrix = np.zeros((len(p0_actions), len(p1_actions)))
        for (p0_index, p0_action) in enumerate(p0_actions):
            for (p1_index, p1_action) in enumerate(p1_actions):
                # make moves
                moves_made = simulate_moves(hands, played, p0_action, p1_action)
                # find payoff of result
                succ_key = tuple(hands[1] + hands[0] + played[0] + played[1])                    
                try:
                    payoff_matrix[p0_index][p1_index] = self.dp[round_num][next_dp_index][succ_key]
                except Exception as e:
                    
                    line = ""
                    for i in range(4):
                        line += str(hash_state[i * 15:i * 15 + 15]) + " "
                    print(line)
                    line = ""
                    for i in range(4):
                        line += str(succ_key[i * 15:i * 15 + 15]) + " "
                    print(line)
                    raise e
                # unmake moves
                unsimulate_moves(hands, played, moves_made)

        # find mixed strats, store if the player whose move we want to find has a mixed equilibrium
        p0_strat = np.asarray(solve_lp(payoff_matrix, p0_actions, p1_actions))
        p1_strat = np.asarray(solve_lp(-1 * np.transpose(payoff_matrix), p1_actions, p0_actions))
        if return_move_player is not None:
            strat_to_save = p0_strat if return_move_player == 0 else p1_strat
            pure_strats = []
            for i in range(len(strat_to_save)):
                if not np.isclose(strat_to_save[i], 0.0):
                    pure_strats.append(i)
            if len(pure_strats) > 1:
                actions_to_save = [legal_moves[return_move_player][strat_index] for strat_index in pure_strats]
                self.collector.save_mixed(actions_to_save, next_dp_index - 1)
                
        payoff = p0_strat @ payoff_matrix @ p1_strat
     
        # return payoff for DP exploration, or pure strat to play if making a move
        if return_move_player is not None:
            if return_move_player == 0:
                p0_pure = sample_from_mixed(p0_strat)
                return p0_actions[p0_pure]
            else:
                p1_pure = sample_from_mixed(p1_strat)
                return p1_actions[p1_pure]
        return payoff

    def play_turn(self, state, player_num):
        """Assumes 2 players"""
        # save state for equilibrium data collection
        self.curr_state = state

        # find move
        if len(state.hands[0]) == 1:
            return [0]
        # fill dp table if not filled yet
        if not self.dp[self.round_num]:
            self.cards_per_round = state.cards_per_round
            puddings_per_round = [state.puddings_in_round(i) for i in range(self.round_num, self.num_rounds)]
            num_puddings = sum(puddings_per_round)
            for round_num in range(self.num_rounds - 1, self.round_num - 1, -1):
                num_puddings -= puddings_per_round.pop()
                # construct starting state from future hands and num puddings
                start_state_list = [0 for i in range(4 * NUM_UNIQUE_CARDS)]
                for p in range(2):
                    for card in state.future_hands[round_num][p]:
                        start_state_list[NUM_UNIQUE_CARDS * p + card_to_index[card]] += 1
                start_state_list[PUDDING_HASH_INDEX[0]] += num_puddings
                start_state_hashes = []
                for i in range(num_puddings + 1):
                    start_state_hashes.append(tuple(start_state_list))
                    start_state_list[PUDDING_HASH_INDEX[0]] -= 1
                    start_state_list[PUDDING_HASH_INDEX[1]] += 1
                self.find_all_states(start_state_hashes, round_num)
                self.fill_dp(round_num)

        action_to_play = self.find_payoff_lp(state.hash_key(), self.turn_num, self.round_num, return_move_player=player_num)
        self.turn_num += 1
        return hash_action_to_index_action(action_to_play, state.hands[player_num])
    
    def round_end(self, round_num):
        self.round_num = round_num + 1
        self.turn_num = 0
        self.dp[round_num].clear()
    