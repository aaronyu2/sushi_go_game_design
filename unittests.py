from dp import DPAgent, index_to_card, expand_hash_state, NUM_UNIQUE_CARDS
from mcts import MinmaxAgent
from state import GameState, SushiCardType

START_STATE = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# START_STATE = (0, 1, 1, 2, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 2, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
NUM_TURNS = sum(START_STATE[:NUM_UNIQUE_CARDS])

def test_dp():
    # set up single round game
    players = [DPAgent(), DPAgent()]
    state = GameState.make_empty(len(players), NUM_TURNS, NUM_TURNS)
    hands, _ = expand_hash_state(START_STATE)
    state.future_hands = [[[] for i in range(2)]]
    for player_num in range(2):
        for i in range(NUM_UNIQUE_CARDS):
            for j in range(hands[player_num][i]):
                state.future_hands[0][player_num].append(index_to_card[i])

    # simulate round
    state.new_round()
    for turn in range(NUM_TURNS):
        to_play = []
        for i, player in enumerate(players):
            played = player.play_turn(state, i)
            if len(played) == 2:
                if state.chopsticks[i] <= 0:
                    raise ValueError(f'ai {player} tried to play 2 cards without chopsticks')
            elif len(played) != 1:
                raise ValueError(f'ai {player} tried to play {len(played)} cards')
            to_play.append(played)
        print(state.hands)
        print(to_play)

        # update game state
        state.make_moves(to_play)
        print(expand_hash_state(state.hash_key()))

if __name__ == '__main__':
    test_dp()
