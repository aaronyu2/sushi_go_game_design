from state import GameState, SushiCardType
from typing import List
from random import randint

class Agent():
    def play_turn(self, state: GameState, player_num: int) -> List[SushiCardType]:
        pass

    def data_to_log(self, file_path, file_num):
        pass

    def round_end(self, round_num):
        pass

class RandomAgent(Agent):
    def play_turn(self, state: GameState, player_num: int) -> List[SushiCardType]:
        size = len(state.hands[player_num])
        idx = randint(0, size - 1)
        return [idx]
    

pref_eval_list = [
    SushiCardType.SQUID_NIGIRI, 
    SushiCardType.SASHIMI, 
    SushiCardType.TEMPURA, 
    SushiCardType.MAKI_ROLLS_3,
    SushiCardType.PUDDING,
    SushiCardType.SALMON_NIGIRI,
    SushiCardType.DUMPLING,
    SushiCardType.MAKI_ROLLS_2,
    SushiCardType.EGG_NIGIRI,
    SushiCardType.MAKI_ROLLS_1,
    SushiCardType.WASABI,
    SushiCardType.CHOPSTICKS,
    ]

class PrefAgent(Agent):
    def play_turn(self, state: GameState, player_num: int) -> List[SushiCardType]:
        best_index, best_val = None, None
        for i in range(len(state.hands[player_num])):
            curr_val = -1 * pref_eval_list.index(state.hands[player_num][i])
            if best_val is None or curr_val > best_val:
                best_val = curr_val
                best_index = i
        return [best_index]

pref2_eval_list = [
    SushiCardType.SQUID_NIGIRI, 
    SushiCardType.SASHIMI, 
    SushiCardType.TEMPURA, 
    SushiCardType.DUMPLING,
    SushiCardType.SALMON_NIGIRI,
    SushiCardType.PUDDING,
    SushiCardType.MAKI_ROLLS_3,
    SushiCardType.MAKI_ROLLS_2,
    SushiCardType.EGG_NIGIRI,
    SushiCardType.MAKI_ROLLS_1,
    SushiCardType.WASABI,
    SushiCardType.CHOPSTICKS,
    ]

class PrefAgent2(Agent):
    def play_turn(self, state: GameState, player_num: int) -> List[SushiCardType]:
        best_index, best_val = None, None
        for i in range(len(state.hands[player_num])):
            curr_val = -1 * pref2_eval_list.index(state.hands[player_num][i])
            if best_val is None or curr_val > best_val:
                best_val = curr_val
                best_index = i
        return [best_index]
    

pref3_eval_list = [
    SushiCardType.SQUID_NIGIRI, 
    SushiCardType.PUDDING,
    SushiCardType.SASHIMI, 
    SushiCardType.TEMPURA, 
    SushiCardType.DUMPLING,
    SushiCardType.MAKI_ROLLS_3,
    SushiCardType.SALMON_NIGIRI,
    SushiCardType.MAKI_ROLLS_2,
    SushiCardType.EGG_NIGIRI,
    SushiCardType.MAKI_ROLLS_1,
    SushiCardType.WASABI,
    SushiCardType.CHOPSTICKS,
    ]

class PrefAgent3(Agent):
    def play_turn(self, state: GameState, player_num: int) -> List[SushiCardType]:
        best_index, best_val = None, None
        for i in range(len(state.hands[player_num])):
            curr_val = -1 * pref3_eval_list.index(state.hands[player_num][i])
            if best_val is None or curr_val > best_val:
                best_val = curr_val
                best_index = i
        return [best_index]