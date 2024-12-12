---------------------------------------------------------------------------
Required packages: scipy, pydantic
Required packages for Jupyter notebooks: pandas, matplotlib, numpy, seaborn
---------------------------------------------------------------------------
To run TestSushi:
./TestSushi -p player0 player1 -n num_games [-s starting_cards] [-r num_rounds] [-c cards_per_round] [-rs random_seed] [-v]

player0, player1: Player type, one of the following:
    random: Random agent.
    pref, pref2, pref3: Preference agent, plays according to a fixed preference order.
    mcts[num_playouts]: Uses random playouts to approximate values of successor states, only looking at single-card plays, and chooses the move with the best worst-case payoff. Example: mcts1000 for 1000 playouts per successor.
    minmax[num_playouts]: Same as MCTS agent, but looks at all legal moves, including chopsticks.
    lp[num_playouts]: Constructs a payoff matrix using random playouts, then solves a linear program for each move to find the optimal mixed strategy given the estimated payoff matrix.
    memo: Uses memoization to recursively find the value of each state starting from the initial state, taking the best worst-case payoff as the payoff of each state.
    dp: Uses dynamic programming to recursively find the value of each state, working backwards from terminal states. Uses linear programming to solve for the optimal mixed strategy at each state and determine the expected payoff.
num_games: Number of games to simulate.
starting_cards: Number of cards to deal to each player at the start of each round.
num_rounds: Number of rounds to play in each game.
cards_per_round: Number of cards each player should play per round, with any unplayed cards being discarded.
random_seed: A random seed, if any, to use.
-v: Include to write the game state to a file "game_states####" after each turn.

Use parameters.py to change scoring rules.
Use data_analysis.ipynb to extract and interpret results.
The following files are for testing code and are not needed for data collection: analyze_states.ipynb, experiments.ipynb, unittests.py

Each time TestSushi is run, a new folder is created in collected_data, and all data is written to files in that folder. Use clear_data.sh to remove the contents of collected_data and reset the filename used for subsequent runs. collected_data is created by calling make if collected_data is not already present.