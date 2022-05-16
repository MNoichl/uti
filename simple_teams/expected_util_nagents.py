import math
import itertools
from typing import List, Any

import numpy as np

# DISCLAIMER: this is very much work in progress


def calculate_utils_n_agents(omega, n, m, game_matrix, player): # omega = tr-probability, n = number agents, m = number action per agent, game_matrix is matrix with r cells and n-tuples per cell
    #all_coordinates_in_game = list(itertools.product(game_matrix.shape, repeat=n)) #
    g = list(itertools.product(*[list(range(x)) for x in game_matrix.shape]))
    all_coordinates_in_game = []
    for profile in g:
        new_profile = profile[:-1]
        if all( i != new_profile for i in all_coordinates_in_game):
            all_coordinates_in_game.append(new_profile)
        else:
            continue
    # print('all coordinates:', all_coordinates_in_game)
    # all_coordinates_in_game = list(itertools.product(*[list(range(x)) for x in game_matrix.shape]))
    #all_coordinates_in_game = list(
    #    itertools.product(*[list(range(x)) for x in game_matrix.shape])
    #) # Why is there no 'repeat' variable? Is this already the correct length?
    a = list(itertools.repeat(m, n)) # creates list of n times number m, indicating how many actions (m) each agent (in total n agents) has
    a.insert(0, m ** n) # adds number m**n on zero's position of a, this is now a tuple indicating how many actions the team and each agent has
    a.append(n)
    shaped_utils = np.zeros(a)  # this is now an n+1 dimensional table in which each cell represents one possible profile; n dimensions have length m (each agent has m actions) the last dimension has length m**n (team has m**n actions)
    for index, team_plays in enumerate(all_coordinates_in_game): # index, list of n action for team reasoners
        team_plays_list = list(team_plays)
        for non_team_cell in all_coordinates_in_game: # list of n actions for individual reasoners
            non_team_cell_list = list(non_team_cell)
        # are team_play and non_team_cell tuples or lists? We need lists! If tuples: convert!
            k = 0
            A = [] # will have 2^n entries, which are n-tuples; all combination of actions when each player is team reasoning or not
            while k <= n:
                sub_profiles = list(itertools.combinations(list(range(len(non_team_cell_list))), k)) # gives all possible sets of indeces of actions from profile non_team_cell with cardinality k
                for subset in sub_profiles: # subset contains indeces of all current team reasoners
                    profile_copy = non_team_cell_list.copy()
                    for j in subset:
                        profile_copy[j] = team_plays_list[j] # substituting individual actions with team actions for team reasoners
                    A.append([profile_copy, k]) # saves profile-combinations with numbers of team reasoners respectively
                k += 1
            util = 0
            # print('all possible profiles:', A)
            for profile in A:  # for one tuple
                # print(type(profile[0]))
                # print('position in game:', game_matrix[tuple(profile[0])])
                # print('util in game:', game_matrix[tuple(profile[0])][0])
                util = util + ((omega ** profile[1]) * ((1 - omega) ** (n - profile[1])) * game_matrix[tuple(profile[0])][player]) # calculating expected utility for team_play and non_team_cells, i.e. considering each profile in A multiplying the probability (depending on how many team reasoners) and its utility
            shaped_utils[index, non_team_cell] = util  # where do I get the index from?
    return shaped_utils # for each profile (indiv, tr) we will have a list of tuples containing profile combination of this profile (depending on whether agents play individual or team strategy) and number of team-reasoners


# game = np.array([[[[3,0,0], [0,0,1]], [[0,0,1], [0,0,1]]], [[[3,0,0], [0,0,1]],[[3,0,0], [0,0,1]]]])
# print(calculate_utils_n_agents(0.6, 3, 2, game, 0))