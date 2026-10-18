"""Self-improving Slay the Spire 2 agent.

Combat is played by information-set MCTS over cloned simulator states; every other
decision (card rewards, map, shop, rest, events, ancients) is chosen by a learned run
value function evaluated on the states each option leads to. Self-play improves the
value function, which improves the decisions, which improves the data.
"""

__version__ = "0.1.0"
