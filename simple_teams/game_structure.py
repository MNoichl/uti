import numpy as np
import pandas as pd
import networkx as nx


import nashpy as nash

from ipysankeywidget import SankeyWidget
from ipywidgets import Layout


from .utils import random_argmax

import collections
import bisect

from .team_reasoning import team_reason

import matplotlib.pyplot as plt

# Just allow n_choices to be a list, for asymmetric games?
class game:

    """An game-class to set up and visualize games.

    Attributes:
        name (str): name of the game. maybe useful to keep track of things.
        n_players (int): number of the players in the game.
        n_choices (int): number of choices for the players. This class currently supports
            games with assymmetric payoffs, but not games with unequal numbers of choices.
            Somebody could implement that.
        payoffs (list): list of lists with length n_players**n_choices.
            Each item of payoffs is a list of the payoffs for each player
            under the a certain configuration. The order is important.
        size (dict): Size of the frame in which the game gets shown. Form: dict(width=2200, height=1570)
    """

    def __init__(
        self,
        name="prisoners_dilemma",  # Standardwerte überdenken...
        n_players=2,
        n_choices=2,
        payoffs=[[2, 2], [0, 0], [0, 0], [1, 1]],
        size=dict(width=2200, height=1570),
    ):
        """[init function of game]

        Args:
            name (str, optional): [The name of the game]. Defaults to "prisoners_dilemma".
            n_choices (int, optional): [ number of the players in the game.]. Defaults to 2.
            payoffs (list, optional): [number of choices for the players. This class currently supports
            games with assymmetric payoffs, but not games with unequal numbers of choices.
            Somebody could implement that, eg. as lists that get passed.]. Defaults to [[2, 2], [0, 0], [0, 0], [1, 1]].
            size ([dict], optional): [Size of the frame in which the game gets shown]. Defaults to dict(width=2200, height=1570).
        """

        self.name = name
        self.n_players = n_players
        self.n_choices = n_choices

        #         self.choice_labels
        self.tree = nx.balanced_tree(
            n_choices, n_players
        )  # This limits us to games with symmetric numbers of choices!
        self.payoffs = payoffs
        self.size = size
        self.path_array = self.get_path_array()

    def show_game(self):
        """
        Plots the defined game as a Sankey diagram.
        """
        T = self.tree
        distances_to_origin = []
        for node in T.nodes():
            distances_to_origin.append(len(nx.shortest_path(T, source=0, target=node)))

        T_edges = T.edges(data=False)
        levels = [distances_to_origin[x[0]] for x in T_edges]

        widths = []
        val = 1
        for k in range(0, self.n_players):
            val = val * self.n_choices
            widths.append(val)

        values = [widths[np.max(levels) - x] for x in levels]
        # values = [(np.max(level) - x) * branches for x in level]

        choice_counter = 1
        payoff_counter = 0
        for node, level in zip(list(T.nodes()), distances_to_origin):
            #     print(node)
            #     print(level-1)
            if level == 1:
                T = nx.relabel_nodes(T, {node: "(" + str(node) + ") init"}, copy=True)
            if level > 1 and level != np.max(distances_to_origin):
                T = nx.relabel_nodes(
                    T,
                    {
                        node: "("
                        + str(node)
                        + ") Player: "
                        + str(level - 1)
                        + ", Choice: "
                        + str(choice_counter)
                    },
                    copy=True,
                )
                #         print('Player: ' + str(level-1)+', Choice: ' + str(choice_counter))
                choice_counter += 1
                if choice_counter > self.n_choices:
                    choice_counter = 1
            if level == np.max(distances_to_origin):
                T = nx.relabel_nodes(
                    T,
                    {
                        node: "("
                        + str(node)
                        + ") Player: "
                        + str(level - 1)
                        + ", Choice: "
                        + str(choice_counter)
                        + " ["
                        + ", ".join([str(x) for x in self.payoffs[payoff_counter]])
                        + "]"
                    },
                    copy=True,
                )
                choice_counter += 1
                payoff_counter += 1
                if choice_counter > self.n_players:
                    choice_counter = 1

        sankey_df = pd.DataFrame(
            np.array(T.edges(data=False)), columns=["source", "target"]
        )
        sankey_df["value"] = values
        sankey_df["color"] = "#1A23408C"

        layout = Layout(
            width=str(self.size["width"]),
            height=str(self.size["height"]),
        )
        display(
            SankeyWidget(
                width=self.size["width"],
                height=self.size["height"],
                links=sankey_df.to_dict("records"),
                margins=dict(top=0, bottom=0, left=50, right=300),
                layout=layout,
            )
        )

    def get_path_array(self):
        paths = []
        for node in self.tree:
            if self.tree.degree(node) == 1:  # it's a leaf
                paths.append(nx.shortest_path(self.tree, 0, node))
        paths = np.array(paths)[:, 1:]

        #         choice_counter = 1
        for column in range(paths.shape[1]):
            unique = np.unique(paths[:, column])
            choice_counter = 1
            for un in unique:
                paths[paths[:, column] == un, column] = choice_counter
                choice_counter += 1
                if choice_counter > self.n_choices:
                    choice_counter = 1
        return paths

    def return_payoffs(self, choices_to_check):

        return self.payoffs[
            np.where(np.all(np.array(choices_to_check) == self.path_array, axis=1))[0][
                0
            ]
        ]

    def return_team_reasoners_choice(self, mode="first"):

        if mode == "first":
            return np.argmax(np.mean(np.array(self.payoffs), axis=1))

        elif mode == "random":
            return random_argmax(np.mean(np.array(self.payoffs), axis=1))

        elif mode == "multiple":
            b = np.mean(np.array(self.payoffs), axis=1)
            return np.flatnonzero(b == np.max(b))

    def return_players_matrix(self, player=0):
        """
        Returns a payoff-matrix for the player, indexed starting with 0.
        Not 100% sure, whether it's already correct...
        """
        return np.array(self.payoffs)[:, player].reshape(
            tuple([self.n_choices] * self.n_players)
        )

    def return_nashpy_equilibrium(self):
        if self.n_players == 2:
            this_game = nash.Game(
                self.return_players_matrix(0), self.return_players_matrix(1)
            )

            return list(this_game.support_enumeration())

        else:
            return "Equilibria for a n_players != 2 are not yet implemented."

    def return_non_trs_choice(self, mode="first"):  # This is bad!
        """
        returns indices of the choice(s) of a non-team-reasoner
        """
        eq = self.return_nashpy_equilibrium()  # this ought to be redone in gambit!

        if isinstance(eq, str):
            return eq
        else:
            # print(eq[np.random.choice(np.arange(len(eq)), 1)[0]])
            eq = eq[np.random.choice(np.arange(len(eq)), 1)[0]][0]
            # print(eq)

            if mode == "first":
                return np.argmax(eq)

            elif mode == "random":
                return random_argmax(eq)

            elif mode == "multiple":
                return np.flatnonzero(b == np.max(eq))

    def set_up_TR_strategies(self, steps=100):
        # print(self.name)
        # Alternative precomputed omega-approach:
        base_space = np.linspace(0.0, 1.0, steps)
        player_data = collections.OrderedDict()
        # present_utils = collections.OrderedDict()

        for omega in base_space:
            players_strategies = team_reason(
                self.return_players_matrix(0), self.return_players_matrix(1), omega
            )
            player_data[omega] = players_strategies
            # present_utils[omega] = exp_utils

        self.player_data = player_data
        # self.present_utils = present_utils

    def plot_TR_utils(self, player=0, figsize=(20, 10)):
        if not hasattr(self, "player_data"):
            self.set_up_TR_strategies()

        my_omegas, my_utils = [], []
        for at_omega in self.player_data:
            # print(self.player_data[at_omega][str(player)])
            for this_util in self.player_data[at_omega][str(player)]["utils"]:
                my_omegas.append(at_omega), my_utils.append(this_util)

        utils_frame = pd.DataFrame([my_omegas, my_utils]).T
        utils_frame.columns = ["omg", "uti"]
        utils_frame["count"] = utils_frame.groupby("omg")["omg"].transform("count")

        fig, ax = plt.subplots(figsize=figsize)
        plt.style.use("opinions.mplstyle")
        plt.scatter(
            utils_frame["omg"],
            utils_frame["uti"],
            s=utils_frame["count"] * 20,
            c="#1a2340",
        )

        ax.set_xlabel("Omega")
        ax.set_ylabel("Expected utility at equilibrium")
        ax.set_title(self.name)

    def return_TR_strategy(self, omega):

        # Fix problem with omega = 0,1
        if not hasattr(self, "player_data"):
            self.set_up_TR_strategies()

        known_omegas = np.array(list(self.player_data.keys()))
        upper = known_omegas[known_omegas > omega].min()
        lower = known_omegas[known_omegas < omega].max()
        upper_strategy = self.player_data[upper]["0"]["strategies"]
        lower_strategy = self.player_data[lower]["0"]["strategies"]

        # print(np.array_equal(upper_strategy, lower_strategy))
        if np.array_equal(upper_strategy, lower_strategy):
            return lower_strategy
        else:
            additional_omega = team_reason(
                self.return_players_matrix(0), self.return_players_matrix(1), omega
            )
            self.player_data[omega] = additional_omega
            return additional_omega["0"]["strategies"]


#######################################                   #################################


class game_mixture:
    # games_ list is a list of tuples

    def __init__(self, games_list):
        self.games_list = [x[0] for x in games_list]
        self.prob_list = [x[1] for x in games_list]
        self.prob_list = self.prob_list / np.sum(self.prob_list)

    def return_game(self):

        return self.games_list[
            np.random.choice(len(self.prob_list), 1, p=self.prob_list)[0]
        ]
