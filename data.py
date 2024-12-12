import pandas as pd
import numpy as np
import os
import random
import pickle
from datetime import datetime
from state import SushiCardType, PUDDING_SCORE

DATA_FOLDER = "collected_data"

enum_to_str = {
    SushiCardType.TEMPURA: 'TEMPURA',
    SushiCardType.SASHIMI: 'SASHIMI',
    SushiCardType.DUMPLING: 'DUMPLING',
    SushiCardType.MAKI_ROLLS_1: 'MAKI_ROLLS_1',
    SushiCardType.MAKI_ROLLS_2: 'MAKI_ROLLS_2',
    SushiCardType.MAKI_ROLLS_3: 'MAKI_ROLLS_3',
    SushiCardType.SALMON_NIGIRI: 'SALMON_NIGIRI',
    SushiCardType.SQUID_NIGIRI: 'SQUID_NIGIRI',
    SushiCardType.EGG_NIGIRI: 'EGG_NIGIRI',
    SushiCardType.PUDDING: 'PUDDING',
    SushiCardType.WASABI: 'WASABI',
    SushiCardType.CHOPSTICKS: 'CHOPSTICKS',
    SushiCardType.HIDDEN: 'HIDDEN'
}

class DataCollector():
    def __init__(self, description, start_hand_size, cards_per_round, num_rounds):
        # self.description = description
        self.start_hand_size = start_hand_size
        self.cards_per_round = cards_per_round
        self.num_rounds = num_rounds
        self.turn_numbers = dict()
        self.lp_pure = [dict() for _ in range(cards_per_round)]
        self.states_to_write = []
        self.mixed = []

        # make new folder numbered by file_num
        with open(f"{DATA_FOLDER}/filename", "r") as f:
            self.file_num = int(f.read())
            self.file_path = f"{DATA_FOLDER}/{self.file_num:04}"
            while os.path.exists(self.file_path):
                self.file_path += str(random.randint(0, 9))
            os.makedirs(self.file_path)

        # update filename for next run
        with open(f"{DATA_FOLDER}/filename", "w") as f:
            f.write(str(self.file_num + 1))

        # write label to folder
        with open(f"{self.file_path}/description{self.file_num:04}", "w") as f:
            f.write(" ".join(description) + "\n" + str(datetime.now()) + "\n")
            f.write(f"PUDDING_SCORE = {PUDDING_SCORE}")


    def save(self, scores_df=None, scores_list=None, player_names=None):
        file_num = self.file_num
        file_path = self.file_path

        # write data to folder
        # write turn numbers
        self.turn_numbers_list = []
        for card_enum in enum_to_str:
            if card_enum in self.turn_numbers:
                self.turn_numbers_list.append([enum_to_str[card_enum], self.turn_numbers[card_enum]])
            else:
                self.turn_numbers_list.append([enum_to_str[card_enum], []])
        with open(f"{file_path}/turn_numbers{file_num:04}.pkl", "wb") as f:
            pickle.dump(self.turn_numbers_list, f)

        # write scores df to csv
        if scores_df is not None:
            scores_df.to_csv(f"{file_path}/scores_df{file_num:04}.csv")
        if scores_list is not None:
            with open(f"{file_path}/scores_list{file_num:04}.pkl", "wb") as f:
                pickle.dump(scores_list, f)
        if player_names is not None:
            with open(f"{file_path}/player_names{file_num:04}.pkl", "wb") as f:
                pickle.dump(player_names, f)

        # write LP agent collected data if it exists
        if self.lp_pure[0]:
            with open(f"{file_path}/lp_pure{file_num:04}.pkl", "wb") as f:
                pickle.dump(self.lp_pure, f)

        

    def handle_error(self):
        self.write_states()

    def add_turn_number(self, card, turn_number):
        if card not in self.turn_numbers:
            self.turn_numbers[card] = [0 for i in range(self.cards_per_round)]
        self.turn_numbers[card][turn_number] += 1

    def save_lp_pure(self, num_pure, state):
        lp_pure_dict = self.lp_pure[state.start_hand_size - len(state.hands[0])]
        if num_pure in lp_pure_dict:
            lp_pure_dict[num_pure] += 1
        else:
            lp_pure_dict[num_pure] = 1

    def save_state(self, state, round_start=False):
        key = state.hash_key()
        if round_start:
            self.states_to_write.append([])
        self.states_to_write[-1].append(key)

    def save_mixed(self, strat, turn_number):
        self.mixed.append([strat, turn_number])

    def write_states(self):
        with open(f"{self.file_path}/game_states{self.file_num:04}", "a") as f:
            f.write(f"{self.states_to_write}\n")
        self.states_to_write.clear()

        if self.mixed:
            with open(f"{self.file_path}/mixed_strats{self.file_num:04}", "a") as f:
                f.write(f"{self.mixed}\n")
            self.mixed.clear()
        
        



