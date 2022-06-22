import numpy as np
import itertools
import pygambit
from fractions import Fraction
import os
from .expected_util_nagents import calculate_utils_n_agents


def team_reasoning_n_agents(m, n, game, omega): # the game input is a list of n matrices, one for each player with dimensions m**n (m number of actions), omega is number indicating team reasoners fraction
    
    coordinates_game = list(itertools.product(*[list(range(x)) for x in game.shape]))
    game_shape = list(game.shape)
    game_shape.pop()
    profiles_game = list(itertools.product(*[list(range(x)) for x in game_shape]))

    utils = []

    for i in range(n+1):
        util_agent_i = calculate_utils_n_agents(omega, n, m, game, i)
        utils.append(util_agent_i)
        # print('util:', utils)
    
    a = list(np.shape(utils[0]))

    g = pygambit.Game.new_table(a) # https://gambitproject.readthedocs.io/en/latest/pyapi.html
    for ix, payoffs in enumerate(utils): # ix takes int value from 0 to n, payoffs
        # print(ix, payoffs)
        for iy, value in np.ndenumerate(payoffs): # iy is a n+1 tuple indicating the played strategies, values are the utils
            g[iy][ix] = Fraction(str(value))#.limit_denominator()

    f = open("current_game.nfg", "w")
    f.write(g.write())
    f.close()
    
    stream = os.popen("gambit-enumpure current_game.nfg -S")
    output = stream.read()
    NEs = output.split("\n")
    
    NEs = [x for x in NEs if len(x) != 0]

    NEs = [x.replace("NE,", "").replace("\n", "") for x in NEs]

    NEs = [[float(x) for x in y.split(",")] for y in NEs]
    
    # print(NEs)
    
    strategies = {}
    for player in range(n+1):
        if player == 0:
            for i in range(1,n+1):
                strategies[str(i)+"_team"] = []
        else:
            strategies[str(player)+"_individual"] =[]
    strategies["profiles"] = []
    strategies["team_utilities"] = []
    
            
    for NE in NEs:
        counter = 0
        profile = []
        for player in range(n+1):
            options = len(g.players[player].strategies)
            strat = NE[counter : counter + options]
            
            if player == 0:
                for i in range(n):
                    strategie_i_tr = [0]*m
                    strategie_i_tr[profiles_game[np.where(strat)[0][0]][i]] = 1
                    strategies[str(i+1)+"_team"].append(strategie_i_tr)
                profile.append(np.where(strat)[0][0])
                
            else:
                strategies[str(player)+"_individual"].append(strat)
                profile.append(np.where(strat)[0][0])
            counter += options
        strategies["profiles"].append(profile)
        strategies["team_utilities"].append(utils[0][tuple(profile)]) # turn profile into tuple such that it can be used as index
        
    
    # print(strategies)
    
    return strategies


