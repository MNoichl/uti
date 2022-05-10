import math
import itertools
import numpy as np

# DISCLAIMER: this is very much work in progress


def calculate_pre_set_n_agents(n, m, game_matrix): # omega = tr-probability, n = number agents, m = number action per agent, game_matrix is matrix with r cells and n-tuples per cell
    all_coordinates_in_game = list(
        itertools.product(*[list(range(x)) for x in game_matrix.shape])
    )
    shaped_profiles = [] # np.zeros(m**(2*n)) # list; entries will be A's; how many A's are there? Should I create a list of zero's rather than an empty list?
    for index, team_plays in enumerate(all_coordinates_in_game): # index, tuple of n action for team reasoners
        for non_team_cell in all_coordinates_in_game: # tuple of n action for non-team reasoners
        # are team_play and non_team_cell tuples or lists? We need lists! If tuples: convert!
            k = 0
            A = [] # will have 2^n entries, which are n-tuples; all combination of actions when each player is team reasoning or not
            while k <= n:
                sub_profiles = list(itertools.combinations(list(range(len((non_team_cell)))), k)) # gives all possible sets of indeces of actions from profile non_team_cell with cardinality k
                for subset in sub_profiles:
                    profile_copy = non_team_cell.copy()
                    for j in subset:
                        profile_copy[j] = team_plays[j]
                    A.append([profile_copy, k])
                k += 1
            shaped_profiles.append(A) # A is a list of tuples containing a n-tuple (profile) and integer (number of team reasoners)
            # PROBLEM! at the moment shaped_profile does not have enough entries, for each new non_team_cell we will substitute the old, as A will be computed under the same index!
    return shaped_profiles # for each profile (indiv, tr) we will have a list of tuples containing profile combination of this profile (depending on whether agents play individual or team strategy) and number of team-reasoners

def calculate_utils_n_agents(omega, n, m, game_matrix): # omega = tr-probability, n = number agents, m = number action per agent, game_matrix is matrix with r cells and n-tuples per cell
    all_coordinates_in_game = list(
        itertools.product(*[list(range(x)) for x in game_matrix.shape])
    )
    shaped_utils = np.zeros((2**n, m, m))
    for index, profile_list in enumerate(calculate_pre_set_n_agents(omega, n, m, game_matrix)): # for a list of tuples
        util = 0
        for profile in profile_list: # for one tuple
            util = util + ((omega**profile[1])*((1-omega)**(n-profile[1]))*game_matrix[profile[0]])
        shaped_utils[index] = util # where do I get the index from?
    return shaped_utils
