import numpy as np
import itertools
import pygambit
from fractions import Fraction
import os
from .expected_util_nagents import calculate_utils_n_agents


# DISCLAIMER: not done

def team_reasoning_n_agents(m, n, game, omega): # the game input is a list of n matrices, one for each player with dimensions m**n (m number of actions), omega is number indicating team reasoners fraction
    coordinates_game = list(itertools.product(*[list(range(x)) for x in game.shape]))
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
    
    stream = os.popen("gambit-enumpure current_game.nfg -S")
    output = stream.read()
    NEs = output.split("\n")
    
    NEs = [x for x in NEs if len(x) != 0]

    NEs = [x.replace("NE,", "").replace("\n", "") for x in NEs]

    NEs = [[float(x) for x in y.split(",")] for y in NEs]
    print(NEs)
    
    ###### the following code tries to replicate what has been done for 2 agents, however there must be a better way #####
    
    # players_strategies = {}
    # for player in range(n+1):
        # players_strategies[str(player)] = {
            # "strategies": [],
            # "utils": [],
        # }
        
    # players_strategies["tr_consistent_payoffs"] = []
   
    # for NE in NEs:
        # counter = 0
        # NE_indices = []
        # for player in range(n+1):
            # n_options = len(g.players[player].strategies)
            # strat = NE[counter : counter + n_options] # slices out n_options many strategies of the current player
            # where_strat = np.where(strat)[0][0] # outputs strategy of current player in pure NE by creating a tuple of an array with indices of all non-zero values in strategy and then calling the first indice in the first array (however, there is only one anyway)
            # if player == 0:
                # players_strategies["tr_consistent_payoffs"].append(
                    # game[coordinates_game[where_strat]] # where_strat is an int, here: the pure strategy the team plays
                    # coordinates_games[where_strat] is a list of int, indicating the profile behind that strategy
                    # game[coordinates_games[where_strat]] is a list of int, indicating the utils of the individual players for this profile
                # )

            # single_strat = np.zeros(game.shape[0])
            # single_strat[np.array(all_coordinates_in_game)[where_strat, player-1]] = 1.0
                # 
                

            # counter += n_options
    
    
    return 'game is now in gambit'


