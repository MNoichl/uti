import numpy as np
import itertools
import pygambit
from fractions import Fraction
import os


def calculate_utils(omega, game_matrix):
    all_coordinates_in_game = list(
        itertools.product(*[list(range(x)) for x in game_matrix.shape])
    )
    shaped_utils = np.zeros((4, 2, 2)) # np.zeros((number of (tr/non_tr)-combinations, number of strategies of player 0,..., number of strategies of player n ))
    count = 0
    for index, team_plays in enumerate(all_coordinates_in_game):

        for non_team_cell in all_coordinates_in_game:
            #             print(non_team_cell)

            this_util = (
                omega ** 2 * game_matrix[team_plays]
                + omega * (1 - omega) * game_matrix[team_plays[0], non_team_cell[1]]
                + (1 - omega) * omega * game_matrix[non_team_cell[0], team_plays[1]]
                + (1 - omega) ** 2 * game_matrix[non_team_cell]
            )
            shaped_utils[index, non_team_cell[0], non_team_cell[1]] = this_util
    return shaped_utils


def team_reason(game_matrix_player_0, game_matrix_player_1, omega): #welche form haben diese Argumente

    # we need this to project  down from the team to the individual:
    all_coordinates_in_game = list(
        itertools.product(*[list(range(x)) for x in game_matrix_player_0.shape])
    )
    print(all_coordinates_in_game)

    p_1 = calculate_utils(omega, game_matrix_player_0)
    p_2 = calculate_utils(omega, game_matrix_player_1)
    team_matrix = (game_matrix_player_0 + game_matrix_player_1) / 2
    t = calculate_utils(omega, team_matrix)
    all_players_actual_utils = [team_matrix, game_matrix_player_0, game_matrix_player_1]
    all_players_utils = [t, p_1, p_2]
    # print(all_players_utils)
    g = pygambit.Game.new_table(p_1.shape)

    # all_players_payoff_array = np.zeros(p_1.shape)
    # print(all_players_payoff_array)
    for ix, players_payoffs in enumerate([t, p_1, p_2]):
        # print(ix, players_payoffs)
        for iy, values in np.ndenumerate(players_payoffs):
            g[iy][ix] = Fraction(str(values))
            # all_players_payoff_array[iy][ix] = all_players_actual_utils[ix][iy]

    # write-game to file
    f = open("current_game.nfg", "w")
    f.write(g.write())
    f.close()

    # solve game externally:
    stream = os.popen("gambit-enumpure current_game.nfg -S")
    output = stream.read()
    NEs = output.split("\n")
    NEs = [x for x in NEs if len(x) != 0]

    NEs = [x.replace("NE,", "").replace("\n", "") for x in NEs]

    NEs = [[float(x) for x in y.split(",")] for y in NEs]
    # print(NEs)
    p_1_utils = []
    p_1_strategies = []

    # return "test", "test"
    players_strategies = {}
    for player in range(3):
        players_strategies[str(player)] = {
            "strategies": [],
            "utils": [],
            # "team_payoff": [],
        }
    players_strategies["tr_consistent_payoffs"] = []
    for NE in NEs:
        counter = 0
        NE_indices = []
        for player in range(3):  # len(g.players)):
            n_options = len(g.players[player].strategies)
            strat = NE[counter : counter + n_options]
            where_strat = np.where(strat)[0][0]
            if player == 0:
                players_strategies["tr_consistent_payoffs"].append(
                    game_matrix_player_0[all_coordinates_in_game[where_strat]]
                )
            # if player == 0:
            #     single_strat = np.zeros(game_matrix.shape[0])
            #     single_strat[np.array(all_coordinates_in_game)[where_strat, 0]] = 1.0
            #     p_1_strategies.append(single_strat)

            single_strat = np.zeros(game_matrix_player_0.shape[0])
            single_strat[np.array(all_coordinates_in_game)[where_strat, 0]] = 1.0

            players_strategies[str(player)]["strategies"].append(single_strat)

            NE_indices.append(where_strat)

            counter += n_options

            # print(NE_indices)
            # print(all_players_utils[player])
        # print("############################")
        for player in range(3):
            # print(all_players_utils[player])
            players_strategies[str(player)]["utils"].append(
                all_players_utils[player][NE_indices[0]][NE_indices[1]][NE_indices[2]]
            )

        # players_strategies["tr_consistent_payoffs"].append(
        #     all_players_actual_utils[1][all_coordinates_in_game[where_strat]]
        # )

        # e_util = p_1[NE_indices[0]][NE_indices[1]][NE_indices[2]]
        # p_1_utils.append(e_util)
        # players_strategies["0"]["utils"].append(e_util)
    # print(players_strategies)

    return players_strategies  # p_1_strategies, p_1_utils
