# Sushi Go Game Implementation Spec

## Base game rules

* Support scoring and playing of all cards in base Sushi Go game, including chopsticks and pudding.
* Game must be a perfect information game, where players can see all hands dealt in every round, both current and future.
* Previously played cards in the same round must be public information. Cards in previous rounds are not required to be stored in the game state, except for puddings.
* Play must be simultaneous, will implement by finding each player's action, one at a time, before updating the state.
  * (Could optimize for time by calculating equilibria for both players together.)

## Game rule modifications

Take the following as optional command-line arguments, with defaults the same as the base game:

* Starting hand size for each player
* Number of rounds
* Number of cards played per round

## AI agent interface

* Game states must include the players' current hands, their previously played cards in the current round, their future hands, and the number of puddings they have already played. Agents will optimize for points and not wins, so states don't need to include players' score in previous rounds.
* Agents must implement play_turn(), which takes in a state and a number that tells the agent which set of cards is their own. It should return a list of indices corresponding to the locations of the card or cards the agent wants to play in their hand. The program should crash if the length of this list corresponds to an illegal move.
  * Chopsticks cannot be used to play other chopsticks.
  * Only one chopsticks card can be used per turn, meaning playing more than 2 cards per turn is impossible.
  * When the order of cards played matters, wasabi is always required to be played first. If two nigiri are played, the higher value nigiri should be played first.

## Game simulation

* The number of games to simulate, the types of agents used, and the parameters for game rules should be passed in as command line arguments.
* The output of simulation should include:
  * Winrates and tie rates of each player
  * Summary statistics on the number of points scored for each game
  * Data on the turn number within rounds that each card type is played. Different variations of the same card type should be treated differently (eg 2 maki should be treated as a different card from 3 maki). This can be stored in a pandas dataframe and saved to an external file.
    * For pudding, the round number it was played in may also need to be stored.
  * Data on how many points each card type scores per round. This should be calculated both as a total and an average per card.
  * Data on the frequency of each possible equilibrium (can also be stored as a pandas dataframe or numpy matrix)
* The exact cards played in a game (for debugging and testing)

## Additional notes

* Only needs to support 2 players (may change later)
* Runtime should be as fast as possible because of the number of rounds that need to be played.
* Games should be parallelized if possible.
