import random
import sys

NUM_ROUNDS = 3
START_HAND_SIZE = 10
# CARDS_PER_ROUND = 10

CHOPSTICKS = 0
WASABI = 1
PUDDING = 2
TEMPURA = 3
SASHIMI = 4
DUMPLING = 5
MAKI_ROLLS_1 = 6
MAKI_ROLLS_2 = 7
MAKI_ROLLS_3 = 8
SALMON_NIGIRI = 9
SQUID_NIGIRI = 10
EGG_NIGIRI = 11
SALMON_WASABI = 12
SQUID_WASABI = 13
EGG_WASABI = 14

starting_dist = {
    CHOPSTICKS: 4,
    WASABI: 6,
    PUDDING: 10,
    TEMPURA: 14,
    SASHIMI: 14,
    DUMPLING: 14,
    MAKI_ROLLS_1: 6,
    MAKI_ROLLS_2: 12,
    MAKI_ROLLS_3: 8,
    SALMON_NIGIRI: 10,
    SQUID_NIGIRI: 5,
    EGG_NIGIRI: 5,
}

starting_deck = []
for card in starting_dist:
    starting_deck.extend([card for i in range(starting_dist[card])])


def swap_indices(lst, i, j):
    lst[i], lst[j] = lst[j], lst[i]

class State():
    def __init__(self, played=None, hands=None):
        if played is None:
            self.played = [[0] * 15, [0] * 15]
        else:
            self.played = played
        if hands is None:
            self.hands = [[0] * 15, [0] * 15]
        else:
            self.hands = hands

    def __hash__(self):
        return hash(tuple(self.played[0] + self.played[1] + self.hands[0] + self.hands[1]))
    
    def legal_actions(self, player_num):
        moves = []
        for i in range(12):
            if self.hands[player_num][i] > 0:
                moves.append([i])
        if self.played[player_num][CHOPSTICKS] > 0:
            for i in range(1, 12):
                if self.hands[player_num][i] > 1:
                    moves.append([i, i])
            for i in range(1, 12):
                for j in range(i + 1, 12):
                    if self.hands[player_num][i] > 0 and self.hands[player_num][j] > 0:
                        moves.append([i, j])
        return moves
    
    def play_card(self, card, player_num):
        self.hands[player_num][card] -= 1
        if self.played[player_num][WASABI] > 0 and (card == EGG_NIGIRI or card == SALMON_NIGIRI or card == SQUID_NIGIRI):
            self.played[player_num][WASABI] -= 1
            self.played[player_num][card] -= 1
            if card == EGG_NIGIRI:
                self.played[player_num][EGG_WASABI] += 1
            elif card == SALMON_NIGIRI:
                self.played[player_num][SALMON_WASABI] += 1
            elif card == SQUID_NIGIRI:
                self.played[player_num][SQUID_WASABI] += 1
        else:
            self.played[player_num][card] += 1
    
    def make_move(self, moves):
        played = [[num for num in sub] for sub in self.played]
        hands = [[num for num in sub] for sub in self.hands]
        for player_num in range(2):
            move = moves[player_num]
            if len(move) == 1:
                hands[player_num][move[0]] -= 1
                played[player_num][move[0]] += 1
            else:
                played[player_num][CHOPSTICKS] -= 1
                hands[player_num][CHOPSTICKS] += 1
                for card in move:
                    played[player_num][card] += 1
                    hands[player_num][card] -= 1
        return State(played, hands[::-1])
    
    def starting_hands(self, cards):
        for player_num in range(2):
            for i in range(START_HAND_SIZE):
                card = cards[(player_num * START_HAND_SIZE) + i]
                self.hands[player_num][card] += 1

    def __str__(self):
        return str(self.hands) + "/" + str(self.played)

    def __eq__(self, other):
        return self.hands[0] == other.hands[0] and self.hands[1] == other.hands[1] and self.played[0] == other.played[0] and self.played[1] == other.played[1]


def simulate_turn(states):
    next_states = set()
    for state in states:
        for move0 in state.legal_actions(0):
            for move1 in state.legal_actions(1):
                new_state = state.make_move([move0, move1])
                next_states.add(new_state)
    # print([str(state) for state in next_states])
    return next_states

def simulate_game(start_hands):
    num_states = []
    states = [State()]
    states[0].starting_hands(start_hands)
    for i in range(START_HAND_SIZE):
        new_states = simulate_turn(states)
        states = new_states
        num_states.append(len(states))
        print(len(states))
    with open("experiments_data/count_states_result", "a") as f:
        f.write(f"{num_states}\n")
    # with open('count_states.txt', 'w') as f:
    #     for state in states:
    #         f.write(f"{state.played}/{state.hands}\n")
    
    
        


def draw_cards():
    return random.sample(starting_deck, START_HAND_SIZE * 2)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        START_HAND_SIZE = int(sys.argv[1])
    simulate_game(draw_cards())
