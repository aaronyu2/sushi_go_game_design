from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum, auto
from random import randint
from parameters import PUDDING_SCORE, NIGIRI_VALS, MAKI_SCORE, SASHIMI_SCORE, TEMPURA_SCORE, SushiCardType
import copy
import itertools



card_to_index = {
    SushiCardType.CHOPSTICKS: 0,
    SushiCardType.WASABI: 1,
    SushiCardType.PUDDING: 2,
    SushiCardType.TEMPURA: 3,
    SushiCardType.SASHIMI: 4,
    SushiCardType.DUMPLING: 5,
    SushiCardType.MAKI_ROLLS_1: 6,
    SushiCardType.MAKI_ROLLS_2: 7,
    SushiCardType.MAKI_ROLLS_3: 8,
    SushiCardType.SALMON_NIGIRI: 9,
    SushiCardType.SQUID_NIGIRI: 10,
    SushiCardType.EGG_NIGIRI: 11,
    SushiCardType.SALMON_WASABI: 12,
    SushiCardType.SQUID_WASABI: 13,
    SushiCardType.EGG_WASABI: 14,
}

TOTAL_CARD_COUNTS = {
    SushiCardType.TEMPURA: 14,
    SushiCardType.SASHIMI: 14,
    SushiCardType.DUMPLING: 14,
    SushiCardType.MAKI_ROLLS_1: 6,
    SushiCardType.MAKI_ROLLS_2: 12,
    SushiCardType.MAKI_ROLLS_3: 8,
    SushiCardType.SALMON_NIGIRI: 10,
    SushiCardType.SQUID_NIGIRI: 5,
    SushiCardType.EGG_NIGIRI: 5,
    SushiCardType.PUDDING: 10,
    SushiCardType.WASABI: 6,
    SushiCardType.CHOPSTICKS: 4,
}
TOTAL_CARDS = sum(TOTAL_CARD_COUNTS.values())

HAND_SIZES = {
    2: 10,
    3: 9,
    4: 8,
    5: 7,
}

def pair_wasabi(cards: List[SushiCardType]) -> List[SushiCardType]:
    pairs: List[SushiCardType] = []
    wasabi = 0
    for card in cards:
        if card == SushiCardType.WASABI:
            wasabi += 1
        elif card in NIGIRI_VALS:
            if wasabi > 0:
                wasabi -= 1
                pairs.append(card)
    return pairs

def count_card_types(cards: List[SushiCardType]) -> Dict[SushiCardType, int]:
    counts = {}
    for card in SushiCardType:
        counts[card] = 0
    for card in cards:
        counts[card] += 1
    return counts

def score_dumplings(dumplings: int) -> int:
    dumplings = min(dumplings, 5)
    return int(dumplings * (dumplings + 1) / 2)

# NOTE: modifies scores
def divide_points(scores: List[int], total:int , players_points: List[int], winning_points: Optional[int]=None):
    if winning_points is None:
        winning_points = max(players_points)
    players = [ i for i in range(len(players_points)) if players_points[i] == winning_points]
    score = int(total / len(players))
    for i in players:
        scores[i] += score

def score_round(played_cards: List[List[SushiCardType]]) -> List[int]:
    scores = [0] * len(played_cards)
    maki_counts = [0] * len(played_cards)
    for i, cards in enumerate(played_cards):
        counts = count_card_types(cards)
        scores[i] += TEMPURA_SCORE * int(counts[SushiCardType.TEMPURA] / 2)
        scores[i] += SASHIMI_SCORE * int(counts[SushiCardType.SASHIMI] / 3)
        scores[i] += score_dumplings(counts[SushiCardType.DUMPLING])
        # Score nigiri
        for card in cards:
            nigiri_val = NIGIRI_VALS.get(card)
            if nigiri_val is not None:
                scores[i] += nigiri_val
        maki_counts[i] += counts[SushiCardType.MAKI_ROLLS_1] * 1
        maki_counts[i] += counts[SushiCardType.MAKI_ROLLS_2] * 2
        maki_counts[i] += counts[SushiCardType.MAKI_ROLLS_3] * 3
    # Score maki
    ordered = sorted(maki_counts)
    if ordered[-1] > 0:
        divide_points(scores, MAKI_SCORE[0], maki_counts, ordered[-1])
        if ordered[-2] > 0:
            divide_points(scores, MAKI_SCORE[1], maki_counts, ordered[-2])
    return scores

def score_pudding(pudding_counts: List[int]) -> List[int]:
    scores = [0] * len(pudding_counts)
    if all([pudding_counts[i] == pudding_counts[i+1] for i in range(len(pudding_counts)-1)]):
        return scores
    
    divide_points(scores, PUDDING_SCORE, pudding_counts)
    if len(pudding_counts) == 2:
        return scores
    min_val = min(pudding_counts)
    divide_points(scores, -PUDDING_SCORE, pudding_counts, min_val)
    return scores

def swap_indices(lst, i, j):
    lst[i], lst[j] = lst[j], lst[i]

class GameState(BaseModel):
    discard_pile: List[SushiCardType]
    played_cards: List[List[SushiCardType]]
    hands: List[List[SushiCardType]]
    future_hands: List[List[List[SushiCardType]]]
    puddings: List[int]
    wasabis: List[int]
    chopsticks: List[int]
    round_num: int
    scores: List[int]
    start_hand_size: int
    cards_per_round: int

    @classmethod
    def make_empty(cls, num_players: int, start_hand_size: int, cards_per_round: int):
        return cls(discard_pile=[],
                   played_cards = [ [] for i in range(num_players) ],
                   hands = [ [] for i in range(num_players) ],
                   puddings = [0] * num_players,
                   wasabis = [0] * num_players,
                   chopsticks = [0] * num_players,
                   round_num = 0,
                   scores = [0] * num_players,
                   future_hands = [[[] for i in range(num_players)]],
                   start_hand_size = start_hand_size,
                   cards_per_round = cards_per_round,
        )
    
    def new_round(self):
        self.hands = copy.deepcopy(self.future_hands[self.round_num])
        self.wasabis = [0] * len(self.scores)
        self.chopsticks = [0] * len(self.scores)

    def make_moves(self, to_play):
        for i in range(len(to_play)):
            if len(to_play[i]) == 2:
                self.chopsticks[i] -= 1
                first_index = min(to_play[i])
                second_index = max(to_play[i])
                swap_indices(self.hands[i], second_index, -1)
                swap_indices(self.hands[i], first_index, -2)
                card2 = self.hands[i].pop()
                card1 = self.hands[i].pop()
                if card2 == SushiCardType.WASABI:
                    card1, card2 = card2, card1
                self.play_card(card1, i)
                self.play_card(card2, i)
                self.hands[i].append(SushiCardType.CHOPSTICKS)
            else:
                swap_indices(self.hands[i], to_play[i][0], -1)
                card = self.hands[i].pop()
                self.play_card(card, i)
        self.rotate_hands()

    def simulate_moves(self, to_play):
        played = []    
        for i in range(len(to_play)):
            if len(to_play[i]) == 2:
                self.chopsticks[i] -= 1
                first_index = min(to_play[i])
                second_index = max(to_play[i])
                swap_indices(self.hands[i], second_index, -1)
                swap_indices(self.hands[i], first_index, -2)
                card2 = self.hands[i].pop()
                card1 = self.hands[i].pop()
                self.play_card(card1, i)
                self.play_card(card2, i)
                self.hands[i].append(SushiCardType.CHOPSTICKS)
                played.append([card1, card2])
            else:
                swap_indices(self.hands[i], to_play[i][0], -1)
                card = self.hands[i].pop()
                self.play_card(card, i)
                played.append([card])
        self.rotate_hands()
        return played
    
    def unsimulate_moves(self, indices, cards):
        """Undoes a move made using simulate_moves, assuming no other changes were made to the state"""
        self.unrotate_hands()
        for i in range(len(indices)):
            if len(indices[i]) == 2:
                first_index = min(indices[i])
                second_index = max(indices[i])
                # remove chopsticks from hand
                self.hands[i].pop()
                # unplay cards
                self.unplay_card(cards[i][1], i)
                self.unplay_card(cards[i][0], i)
                self.hands[i].append(cards[i][1])
                self.hands[i].append(cards[i][0])
                # unswap hand indices
                swap_indices(self.hands[i], first_index, -2)
                swap_indices(self.hands[i], second_index, -1)
                self.chopsticks[i] += 1
            else:
                self.unplay_card(cards[i][0], i)
                self.hands[i].append(cards[i][0])
                swap_indices(self.hands[i], indices[i][0], -1)

    def play_card(self, card, player_num):
        if card == SushiCardType.PUDDING:
            # uncomment to include pudding in played_cards
            # self.played_cards[player_num].append(SushiCardType.PUDDING)
            self.puddings[player_num] += 1
        elif card == SushiCardType.CHOPSTICKS:
            self.chopsticks[player_num] += 1
        elif card == SushiCardType.WASABI:
            # self.played_cards[player_num].append(SushiCardType.WASABI)
            self.wasabis[player_num] += 1
        elif card == SushiCardType.EGG_NIGIRI:
            if self.wasabis[player_num] > 0:
                self.wasabis[player_num] -= 1
                self.played_cards[player_num].append(SushiCardType.EGG_WASABI)
            else:
                self.played_cards[player_num].append(SushiCardType.EGG_NIGIRI)
        elif card == SushiCardType.SALMON_NIGIRI:
            if self.wasabis[player_num] > 0:
                self.wasabis[player_num] -= 1
                self.played_cards[player_num].append(SushiCardType.SALMON_WASABI)
            else:
                self.played_cards[player_num].append(SushiCardType.SALMON_NIGIRI)
        elif card == SushiCardType.SQUID_NIGIRI:
            if self.wasabis[player_num] > 0:
                self.wasabis[player_num] -= 1
                self.played_cards[player_num].append(SushiCardType.SQUID_WASABI)
            else:
                self.played_cards[player_num].append(SushiCardType.SQUID_NIGIRI)
        else:
            self.played_cards[player_num].append(card)

    def unplay_card(self, card, player_num):
        if card == SushiCardType.PUDDING:
            # uncomment to include pudding in played_cards
            # self.played_cards[player_num].append(SushiCardType.PUDDING)
            self.puddings[player_num] -= 1
        elif card == SushiCardType.CHOPSTICKS:
            self.chopsticks[player_num] -= 1
        elif card == SushiCardType.WASABI:
            # self.played_cards[player_num].append(SushiCardType.WASABI)
            self.wasabis[player_num] -= 1
        elif card == SushiCardType.EGG_NIGIRI:
            if self.played_cards[player_num][-1] == SushiCardType.EGG_WASABI:
                self.wasabis[player_num] += 1
            self.played_cards[player_num].pop()
        elif card == SushiCardType.SALMON_NIGIRI:
            if self.played_cards[player_num][-1] == SushiCardType.SALMON_WASABI:
                self.wasabis[player_num] += 1
            self.played_cards[player_num].pop()
        elif card == SushiCardType.SQUID_NIGIRI:
            if self.played_cards[player_num][-1] == SushiCardType.SQUID_WASABI:
                self.wasabis[player_num] += 1
            self.played_cards[player_num].pop()
        else:
            self.played_cards[player_num].pop()


    def rotate_hands(self):
        self.hands = self.hands[1:] + self.hands[:1]

    def unrotate_hands(self):
        self.hands = self.hands[-1:] + self.hands[:-1]

    def legal_actions(self, player_num):
        if len(self.hands[player_num]) == 1:
            return [[0]]
        
        counts = dict()
        for i in range(len(self.hands[player_num])):
            card = self.hands[player_num][i]
            if card in counts:
                counts[card].append(i)
            else:
                counts[card] = [i]

        single_moves = [[counts[card][0]] for card in counts]
        # if no chopsticks, only return individual cards
        if self.chopsticks[player_num] == 0:
            return single_moves
        
        # if chopsticks, return all 1 card and 2 card combinations
        double_iter = itertools.combinations([move[0] for move in single_moves], 2)
        double_moves = [[move[0], move[1]] for move in double_iter]

        for card in counts:
            if len(counts[card]) > 1:
                double_moves.append((counts[card][0], counts[card][1]))
        return single_moves + double_moves

    def pretty_print(self):
        print(f'ROUND: {self.round_num}')
        for i in range(len(self.played_cards)):
            print(f'PLAYER {i}')
            print(f'\tSCORE: {self.scores[i]}')
            print(f'\tPUDDINGS: {self.puddings[i]}')
            print(f'\tCHOPSTICKS: {self.chopsticks[i]}')
            print(f'\tWASABIS: {self.wasabis[i]}')
            print('\tHAND')
            for card in self.hands[i]:
                print(f'\t\t{card.value}')  
            print('\tPLAYED')
            for card in self.played_cards[i]:
                print(f'\t\t{card.value}')    

    def is_terminal(self):
        return len(self.hands[0]) == self.start_hand_size - self.cards_per_round
    
    def score_played(self):
        return score_round(self.played_cards)

    def get_counts(self):
        num_players = len(self.hands)
        hand_counts = [[0 for _ in range(len(card_to_index))] for _ in range(num_players)]
        played_counts = [[0 for _ in range(len(card_to_index))] for _ in range(num_players)]
        for i in range(num_players):
            for card in self.hands[i]:
                hand_counts[i][card_to_index[card]] += 1
            played_counts[i][card_to_index[SushiCardType.CHOPSTICKS]] += self.chopsticks[i]
            played_counts[i][card_to_index[SushiCardType.WASABI]] += self.wasabis[i]
            played_counts[i][card_to_index[SushiCardType.PUDDING]] += self.puddings[i]
            for played in self.played_cards[i]:
                played_counts[i][card_to_index[played]] += 1
        return hand_counts, played_counts
    
    def puddings_in_round(self, round_num):
        count = 0
        for hand in self.future_hands[round_num]:
            for card in hand:
                if card == SushiCardType.PUDDING:
                    count += 1
        return count
    
    def hash_key(self):
        hands, played = self.get_counts()
        if len(hands) == 2:
            return tuple(hands[0] + hands[1] + played[0] + played[1])
        raise NotImplementedError("Doesn't support more than 2 players yet")
        
    def __eq__(self, other):
        this_hands, this_played = self.get_counts()
        other_hands, other_played = other.get_counts()
        return this_hands == other_hands and this_played == other_played
    
        
        

                
class GameStateSet(BaseModel):
    states: List[GameState]

def get_shuffled_cards() -> List[SushiCardType]:
    cards: List[SushiCardType] = []
    shuffled: List[SushiCardType] = []
    for card, count in TOTAL_CARD_COUNTS.items():
        cards += [card] * count
    for i in range(TOTAL_CARDS - 1, -1, -1):
        idx = randint(0, i)
        shuffled.append(cards.pop(idx))
    return shuffled
