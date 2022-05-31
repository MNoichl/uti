import numpy as np
import itertools
import pygambit
from fractions import Fraction
import os
from .expected_util_nagents import calculate_utils_n_agents

def team_reasoning_n_agents(m, n, game, omega): # the game input is a list of n matrices, one for each player with dimensions m**n (m number of actions), omega is number indicating team reasoners fraction
    utils = []
    i = 0 # looping thorugh all agents
    team_util = 0
    while i < n:
        util_agent_i = calculate_utils_n_agents(omega, n, m, game, i)
        team_util = team_util + util_agent_i # summing over all utils of all agents, to build average
        utils.append(util_agent_i)
        i += 1
    utils.insert(0, team_util / n) # list of all utils, teams at position 0, agent i's on position i (starting from 1)
    # print(np.shape(utils))
    # print(utils)
    a = list(np.shape(team_util))
    # print(a)
    # print(np.shape(utils))
    # print(len(utils))
    # print(utils)
    # a = list(itertools.repeat(m, n+1)) # listing n times the number m
    g = pygambit.Game.new_table(a) # https://gambitproject.readthedocs.io/en/latest/pyapi.html
    for ix, payoffs in enumerate(utils): # ix takes int value from 0 to n, payoffs
        # print(ix, payoffs)
        for iy, value in np.ndenumerate(payoffs): # iy is a n+1 tuple indicating the played strategies, values are the utils
            # print(value)
            g[iy][ix] = Fraction(str(value))

    f = open("current_game.nfg", "w")
    f.write(g.write())
    f.close()

    return 'game is now in gambit'


