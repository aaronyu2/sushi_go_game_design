import argparse
from state import get_shuffled_cards, HAND_SIZES, SushiCardType, GameState, score_round, score_pudding
from data import DataCollector
from typing import List, Optional, Callable, Dict
import random
import sys

import agent
from mcts import MCTSAgent, MinmaxAgent
from lp import LPAgent
from dp import DPAgent
from memo import MemoAgent

def sum_lists(first: List[int], second: List[int]) -> List[int]:
    return [x + y for x, y in zip(first, second)]

def required_length(nmin: int, nmax: Optional[int]):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if (nmin is not None and nmin > len(values)) or (nmax is not None and nmax < len(values)):
                msg = 'argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest, nmin=nmin, nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength

# NOTE: Modifies draw_cards
def deal_hands(draw_cards:List[SushiCardType], player_count:int, cards_per_hand:int) -> List[List[SushiCardType]]:
    hands: List[List[SushiCardType]] = [ [] for i in range(player_count) ]
    for i in range(player_count):
        for _ in range(cards_per_hand):
            hands[i].append(draw_cards.pop())
    return hands

AI_TYPES = {
    "random": agent.RandomAgent,
    "pref": agent.PrefAgent,
    "pref2": agent.PrefAgent2,
    "pref3": agent.PrefAgent3,
}

def make_player(player_type, collector, num_rounds):
    if player_type[:2] == "dp":
        if player_type[:5] == "dpwin":
            return DPAgent(collector=collector, num_rounds=num_rounds, is_score=False)
        return DPAgent(collector=collector, num_rounds=num_rounds, is_score=True)
    if player_type[:4] == "memo":
        return MemoAgent()
    if player_type[:6] == "minmax":
        if len(player_type) == 6:
            return MinmaxAgent(1)
        num_playouts = int(player_type[6:])
        return MinmaxAgent(num_playouts)
    if player_type[:4] == "mcts":
        if len(player_type) == 4:
            return MCTSAgent(1)
        num_playouts = int(player_type[4:])
        return MCTSAgent(num_playouts)
    if player_type[:2] == "lp":
        if len(player_type) == 2:
            return LPAgent(1, collector)
        num_playouts = int(player_type[2:])
        return LPAgent(num_playouts, collector)
    if player_type in AI_TYPES:
        return AI_TYPES[player_type]()
    raise ValueError("Invalid player type")

# Players pass hands to the next higher numbered player wrapping at the top
def run_game(players_str: List[str], verbose: bool, start_hand_size: int, cards_per_round: int, num_rounds: int, collector: DataCollector) -> GameState:
    players = [make_player(player_type, collector, num_rounds) for player_type in players_str]
    draw_cards = get_shuffled_cards()
    state = GameState.make_empty(len(players), start_hand_size, cards_per_round)
    try:
        state.future_hands = [deal_hands(draw_cards, len(players), start_hand_size) for round in range(num_rounds)]
    except IndexError as e:
        print("Not enough cards, increase deck size or decrease cards per hand/number of rounds")
        exit(1)

    for round_num in range(num_rounds):
        state.round_num = round_num
        state.new_round()
        if verbose:
            collector.save_state(state, round_start=True)
        for turn in range(cards_per_round):
            to_play = []
            for i, player in enumerate(players):
                played = player.play_turn(state, i)
                if len(played) == 2:
                    if state.chopsticks[i] <= 0:
                        raise ValueError(f'ai {player} tried to play 2 cards without chopsticks')
                elif len(played) != 1:
                    raise ValueError(f'ai {player} tried to play {len(played)} cards')
                to_play.append(played)

            # update collector with turn numbers
            for player_num in range(len(to_play)):
                for card_index in to_play[player_num]:
                    collector.add_turn_number(state.hands[player_num][card_index], turn)

            # update game state
            state.make_moves(to_play)

            # save state to collector
            if verbose:
                collector.save_state(state, round_start=False)
        round_scores = score_round(state.played_cards)
        state.scores = sum_lists(round_scores, state.scores)
        if verbose:
            collector.save_state(state)

        for i in range(len(players)):
            state.played_cards[i] = [] 
            state.hands[i] = []
            state.wasabis[i] = 0
            state.chopsticks[i] = 0
            players[i].round_end(round_num)

    pudding_scores = score_pudding(state.puddings)
    state.scores = sum_lists(pudding_scores, state.scores)

    if verbose:
        collector.write_states()
    
    return state
    
    


def main(argv, players:List[str], games: int, verbose: bool, start_hand_size: int, cards_per_round: Optional[int], num_rounds: int):
    if cards_per_round is None:
        cards_per_round = start_hand_size
    collector = DataCollector(argv, start_hand_size, cards_per_round, num_rounds)
    scores = []
    names = [ f'{player}_{i}' for i, player in enumerate(players)]
    try:
        for i in range(games):
            result = run_game(players, verbose, start_hand_size, cards_per_round, num_rounds, collector)
            scores.append(result.scores)
    except Exception as e:
        collector.handle_error()
        raise e
        
    if len(players) == 2:
        # df = pd.DataFrame(data=scores, columns=names)
        # df["diff"] = df[names[0]] - df[names[1]]
        # df["p1_win"] = df.apply(diff_to_win, axis=1)
        # print(df.describe())

        p1_wins = p2_wins = ties = 0
        for p1_score, p2_score in scores:
            if p1_score > p2_score:
                p1_wins += 1
            elif p1_score < p2_score:
                p2_wins += 1
            else:
                ties += 1
        print(sys.argv)
        print(f"{names[0]} winrate: {p1_wins / games:.4f}")
        print(f"{names[1]} winrate: {p2_wins / games:.4f}")
        print(f"Tie rate: {ties / games:.4f}")
        
        collector.save(scores_list=scores, player_names=names)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a series of games between a set of AIs")
    parser.add_argument('-p', '--players', nargs='+',
                        help='<Requires 2-5> AI types for players', action=required_length(2, 5), required=True)
    parser.add_argument("-n", type=int, default=100,
                        help="Number of games to run")
    parser.add_argument("-rs", type=int, default=None,
                        help="Set random seed")    
    parser.add_argument("-s", type=int, default=10,
                        help="Starting hand size")
    parser.add_argument("-c", type=int, default=None,
                        help="Cards per round")
    parser.add_argument("-r", type=int, default=3,
                        help="Number of rounds")
    parser.add_argument("-v", action="store_true",
                        help="Write game state to file after each turn")                
    try:
        args = parser.parse_args()
    except argparse.ArgumentTypeError as err:
        print(err)
        exit(1)
    random.seed(args.rs)
    main(sys.argv, args.players, args.n, args.v, args.s, args.c, args.r)
