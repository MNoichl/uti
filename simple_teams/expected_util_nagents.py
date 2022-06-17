import math
import itertools
from typing import List, Any

import numpy as np


def calculate_utils_n_agents(omega, n, m, game_matrix, player): # omega = tr-probability, n = number agents, m = number action per agent, game_matrix is matrix with r cells and n-tuples per cell
    #all_coordinates_in_game = list(itertools.product(game_matrix.shape, repeat=n))
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
    
    team_game = game_matrix
    t = np.expand_dims(np.zeros(list(itertools.repeat(m, n))), axis=n)
    # print('t', t)
    # print('t-shape', t.shape)
    # print(np.concatenate([t, team_game]))
    # print(team_game)
    for profile in all_coordinates_in_game:
        team_util = 0
        for i in range(n):
            team_util += game_matrix[tuple(profile)][i]
        team_util = team_util/n
        # print(team_util)
        # team_game[tuple(profile)].append(team_util) #I want to append "team_util" within team_game[tuple(profile)]
        # well, actually I do not want to append but put it at the first position, hence if profile is (1,2,1) I want to add 4/3 at position 0, to get (4/3, 1, 2, 2)
        # TODO: create np.array with team utils as elements and then stack to game
        # print(team_game[tuple(profile)])
        # print('game shape', team_game.shape)
        # team_game[tuple(profile)[n] = team_util # why is this not working?
        # print(team_game[tuple(profile)])
    # print('new game:', team_game)
    
    a = list(itertools.repeat(m, n)) # creates list of n times number m, indicating how many actions (m) each agent (in total n agents) has
    a.insert(0, m ** n) # adds number m**n on zero's position of a, this is now a tuple indicating how many actions the team and each agent has
    shaped_utils = np.zeros(a)  # this is now an n+1 dimensional table in which each cell represents one possible profile; n dimensions have length m (each agent has m actions) the last dimension has length m**n (team has m**n actions)
    # print(np.shape(shaped_utils))
    # print(shaped_utils)
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
                    A.append([profile_copy, k, index, non_team_cell]) # saves profile-combinations with numbers of team reasoners respectively
                k += 1

                
                
            util = 0
            # print('all possible profiles:', A)
            for profile in A:  # for one tuple
                # print(type(profile[0]))
                # print('position in game:', game_matrix[tuple(profile[0])])
                # print('util in game:', game_matrix[tuple(profile[0])][player])
                util = util + ((omega ** profile[1]) * ((1 - omega) ** (n - profile[1])) * game_matrix[tuple(profile[0])][player]) # ToDo: This is where we should insert a team utility function in case player = team 
                # calculating expected utility for team_play and non_team_cells, i.e. considering each profile in A multiplying the probability (depending on how many team reasoners) and its utility
                # print('probability:', ((omega ** profile[1]) * ((1 - omega) ** (n - profile[1]))))
                # print('util-piece:', util)
            # print(util)
            # print('position 1:',A[0][2])
            # print('position 2:',A[0][3])
            shaped_utils[A[0][2]][A[0][3]] = util  # index = played strategy by team 0 to 7, non_team_cell = played strategies by individuals; together building the index for the current utility in the (8, 2, 2, 2) shape
            # print('where I put it:', shaped_utils[A[0][2], A[0][3]])
            # print('corrected:', shaped_utils[A[0][2]][A[0][3]])
            # print('where it should be:', shaped_utils[0,0,1,1])
    # print('position 0,0,1,1:', shaped_utils[0, 0, 1, 1 ])
    # print('shaped utils:', shaped_utils)
    return shaped_utils # for each profile (indiv, tr) we will have a list of tuples containing profile combination of this profile (depending on whether agents play individual or team strategy) and number of team-reasoners


# game = np.array([[[[3,0,0], [0,0,1]], [[0,0,1], [0,0,1]]], [[[3,0,0], [0,0,1]],[[3,0,0], [0,0,1]]]])
# print(calculate_utils_n_agents(0.6, 3, 2, game, 0))