from state import GameState, SushiCardType, score_round, score_pudding
from agent import Agent
import random
import copy
import numpy as np

def random_playout_round(start_hands, num_not_played, played_cards, puddings):
    """Returns score from randomly playing out the round, and changes puddings"""
    num_left = len(start_hands[0]) - num_not_played
    samples = [random.sample(hand, num_left) for hand in start_hands]
    p1_added = samples[0][:(num_left + 1) // 2] + samples[1][(num_left + 1) // 2:]
    p2_added = samples[1][:(num_left + 1) // 2] + samples[0][(num_left + 1) // 2:]
    for card in p1_added:
        if card == SushiCardType.PUDDING:
            puddings[0] += 1
    for card in p2_added:
        if card == SushiCardType.PUDDING:
            puddings[1] += 1
    return score_round([played_cards[0] + p1_added, played_cards[1] + p2_added])

def random_playout_score(state, play_future_hands=True):
    """Returns the P1 - P2 score difference of one random playout, assumes no chopsticks are used"""
    # tries to minimize copying
    future_hands = state.future_hands
    puddings = [count for count in state.puddings]
    scores = random_playout_round(state.hands, state.start_hand_size - state.cards_per_round, state.played_cards, puddings)
    if play_future_hands:
        for hand_list in future_hands:
            future_scores = random_playout_round(hand_list, state.start_hand_size - state.cards_per_round, [[], []], puddings)
            for i in range(2):
                scores[i] += future_scores[i]
    pudding_scores = score_pudding(puddings)
    for i in range(2):
        scores[i] += pudding_scores[i]
    return scores[0] - scores[1]
    
def eval_random_playout(state, player_num, num_playouts):
    """Estimates the value of a game state by averaging across random playouts"""
    score_diff = 0
    for i in range(num_playouts):
        score_diff += random_playout_score(state)
    if player_num == 1:
        score_diff *= -1
    return score_diff / num_playouts

class MCTSAgent(Agent):
    def __init__(self, num_playouts=1):
        self.num_playouts = num_playouts

    def play_turn(self, state, player_num):
        return self.play_turn_sim(state, player_num)
    
    def play_turn_with_copy(self, state, player_num):
        """Chooses a move to play by simulating each possible next move using a copy of state"""
        best_payoff = None
        best_action = None
        for my_action in range(len(state.hands[0])):
            # determine what opponent's best action is given that you choose my_action
            min_payoff = None
            # min_action = None
            for their_action in range(len(state.hands[0])):
                new_state = copy.deepcopy(state)
                if player_num == 0:
                    new_state.make_moves([[my_action], [their_action]])
                else:
                    new_state.make_moves([[their_action], [my_action]])
                state_value = eval_random_playout(new_state, player_num, self.num_playouts)
                if min_payoff is None or state_value < min_payoff:
                    min_payoff = state_value
                    # min_action = their_action
            # if opponent's best action is still better for you, update your action
            if best_payoff is None or min_payoff > best_payoff:
                best_payoff = min_payoff
                best_action = my_action
        return [best_action]
    
    def play_turn_sim(self, state, player_num):
        """Chooses a move to play by simulating each possible next move using simulate and unsimulate"""
        best_payoff = None
        best_action = None
        for my_action in range(len(state.hands[0])):
            # determine what opponent's best action is given that you choose my_action
            min_payoff = None
            # min_action = None
            for their_action in range(len(state.hands[0])):
                if player_num == 0:
                    moves = [[my_action], [their_action]]
                else:
                    moves = [[their_action], [my_action]]
                cards_played = state.simulate_moves(moves)
                state_value = eval_random_playout(state, player_num, self.num_playouts)
                state.unsimulate_moves(moves, cards_played)
                if min_payoff is None or state_value < min_payoff:
                    min_payoff = state_value
                    # min_action = their_action
                # state.pretty_print()
            # if opponent's best action is still better for you, update your action
            if best_payoff is None or min_payoff > best_payoff:
                best_payoff = min_payoff
                best_action = my_action
        return [best_action]
    
class MinmaxAgent(Agent):
    """Best worst case agent that looks at all possible moves (including chopsticks plays) and uses random playouts to estimate payoffs"""
    def __init__(self, num_playouts=1):
        self.num_playouts = num_playouts

    def play_turn(self, state, player_num):
        """Assumes 2 players"""
        my_actions = state.legal_actions(player_num)
        their_actions = state.legal_actions(1 - player_num)
        best_payoff = None
        best_action = None
        for my_action in my_actions:
            min_payoff = None
            for their_action in their_actions:
                if player_num == 0:
                    moves = [my_action, their_action]
                else:
                    moves = [their_action, my_action]
                cards_played = state.simulate_moves(moves)
                state_value = eval_random_playout(state, player_num, self.num_playouts)
                state.unsimulate_moves(moves, cards_played)
                if min_payoff is None or state_value < min_payoff:
                    min_payoff = state_value
            if best_payoff is None or min_payoff > best_payoff:
                best_payoff = min_payoff
                best_action = my_action
        return best_action
        
            


