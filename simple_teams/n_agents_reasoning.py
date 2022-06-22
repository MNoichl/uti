import numpy as np
import itertools
import pygambit
from fractions import Fraction
import os
from .expected_util_nagents import calculate_utils_n_agents


def team_reasoning_n_agents(m, n, game, omega): 
    
    """Function calculating equilibria
    
    Attributes:
        m (int): number of strategies per agent
        n (int): number of individual agents
        game (numpy array): n+1 dimensions of shape (m,...,m,n)
        omega (float): team reasoning probability
        
    Output:
        dict | with 2n+2 entries
            - i_team (for each individual agent i): list of m-tuples indicating strategies played in the equilibria as a team reasoner
            - i_individual (for each individual agent i): list of m-tuples indicating strategies played in the equilibria as an individual reasoner
            - profiles: list of n+1-tuples, indicating the strategies played by team and individual agents in the equilibria
            - team_utilities: list of floars, indicating the expected utilities of the team in the equilibria
   
   How it works:
       for the team and each individual agent we take the output of the "calculate_utils_n_agents"-function and create a new numpy array of n+2 dimensions
       first dimension has n+1 entries, for team (position 0) and each individual agent (position i for agent i) including the result of the "calculate_utils_n_agents"-function (which is again an n+1 dimensional array)
       then we input this into gambit and let it compute all pure Nash Equilibria for this game; for details see https://gambitproject.readthedocs.io/en/latest/pyapi.html
       after this, we arrange the output as described above in "Output"
   
    """
    
    
    coordinates_game = list(itertools.product(*[list(range(x)) for x in game.shape]))
    game_shape = list(game.shape)
    game_shape.pop()
    profiles_game = list(itertools.product(*[list(range(x)) for x in game_shape]))

    utils = []

    for i in range(n+1):
        util_agent_i = calculate_utils_n_agents(omega, n, m, game, i)
        utils.append(util_agent_i)
    
    a = list(np.shape(utils[0]))

    g = pygambit.Game.new_table(a) 
    for ix, payoffs in enumerate(utils):
        for iy, value in np.ndenumerate(payoffs):
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
        strategies["team_utilities"].append(utils[0][tuple(profile)]) 
    
    return strategies


