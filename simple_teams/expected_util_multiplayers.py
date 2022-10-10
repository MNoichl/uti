import math
import itertools
from typing import List, Any

import numpy as np


def calculate_utils_n_agents(omega, n, m, game_matrix, player):
    
    """Function to compute expected utilities
    
    Attributes:
        omega (float): team reasoning probability
        n (int): number of individual agents
        m (int): number of strategies per agent (at the moment this is equal for all individual agents)
        game_matrix (numpy array): n+1 dimensions of shape (m,...,m,n)
        player (int): \in {0,...,n} iff 0 then the teams utility gets calculated, if i then i'th's players utility gets calculated
        
    Output:
        numpy array | n+1 dimensions of shape (m^n,m ,...,m) each dimension represents choice of player, in particular 0th dimension represents choice of team
        
    How it works:
        - all_coordinates_in_game: we generate an array of all pure profiles for the individual agents
        - teams_utilwe generate an array for the teams utility
        - team_game: we generate a new game matrix with the teams payoff by concetanating the teams_util into the game_matrix
        - A: for each profile (including strategy for team and individual players) we generate a list including the following data
            - profile_copy: possible that could be played based on this profile (since team reasoning is circumspect)
            - k: number of agents who actually team reasoned
            - index: position of the team strategy
            - non_team_cell: position of the individual player's profile
        - util: this is how we calculate the expected utility based on Bacharach (1999)
        
    """
    
    g = list(itertools.product(*[list(range(x)) for x in game_matrix.shape]))
    all_coordinates_in_game = []
    for profile in g:
        new_profile = profile[:-1]
        if all( i != new_profile for i in all_coordinates_in_game):
            all_coordinates_in_game.append(new_profile)
        else:
            continue
    
    teams_util = np.expand_dims(np.zeros(list(itertools.repeat(m, n))), axis=n)

    for profile in all_coordinates_in_game:
        team_util = 0
        for i in range(n):
            team_util += game_matrix[tuple(profile)][i]
        team_util = team_util/n
        teams_util[tuple(profile)] = team_util
    team_game = np.concatenate((teams_util, game_matrix), axis=n, out=None, dtype=None, casting="same_kind") 
    
    a = list(itertools.repeat(m, n))
    a.insert(0, m ** n)
    shaped_utils = np.zeros(a)
    
    for index, team_plays in enumerate(all_coordinates_in_game):
        team_plays_list = list(team_plays)
        for non_team_cell in all_coordinates_in_game:
            non_team_cell_list = list(non_team_cell)
            k = 0
            A = []
            while k <= n:
                sub_profiles = list(itertools.combinations(list(range(len(non_team_cell_list))), k))
                for subset in sub_profiles:
                    profile_copy = non_team_cell_list.copy()
                    for j in subset:
                        profile_copy[j] = team_plays_list[j] 
                    A.append([profile_copy, k, index, non_team_cell]) 
                k += 1

            util = 0

            for profile in A:  
                util = util + ((omega ** profile[1]) * ((1 - omega) ** (n - profile[1])) * team_game[tuple(profile[0])][player]) 
                
            shaped_utils[A[0][2]][A[0][3]] = util 
            
    return shaped_utils 