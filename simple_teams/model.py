from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.datacollection import DataCollector


import numpy as np
import networkx as nx
import random
from copy import deepcopy

from .learners import bayesian_beta_learner
from .learners import bayesian_gaussian_process

from .team_reasoning import team_reason
from .utils import random_argmax
from .game_structure import game, game_mixture

from functools import partial

import collections


class MultiAgentActivation(BaseScheduler):
    """Broken out from mesa, same as random activation.
    We might want to mess with this later..."""

    def step(self) -> None:
        """Executes the step of all agents, one at a time, in
        random order.
        """
        for agent in self.agent_buffer(shuffled=True):
            agent.step()
        self.steps += 1
        self.time += 1


class MyAgent(Agent):

    """An agent for a mesa simulation.

    Attributes:
        alive (bool): is the agent alive? (currently unused.)
        gathered_utility (int): How much payoff has the agent gathered in the games it has played so far?
        probability_team_reasoning (float or int): Number between 0 and one: How likely is an agent to
            team-reason, given that they are a team-reasoner.
        team_reasoner (bool): Is the agent a team-reasoner?
        team_reasoning_counter (int): How often has the agent engaged in team reasoning so far.
    """

    def __init__(
        self, unique_id, model, team_reasoner=False, probability_team_reasoning=0.9
    ):
        """Constructor method"""
        super().__init__(unique_id, model)

        self.team_reasoner = team_reasoner
        self.alive = True
        self.gathered_utility = 0
        self.probability_team_reasoning = probability_team_reasoning  # strictly speaking unnecesary, lets use to keep track!
        self.team_reasoning_counter = 0

        self.my_learner = deepcopy(model.learner)

    def step(self):
        """Step function for the agents: First we choose another neighboring agent to play with. Both agents then reason
        about the game, and play accordingly. Finally the current agent collects their payoff, depending on the
        outcome."""
        played_game = self.model.game_generator()
        # print(played_game.name)
        my_neighbors_ids = list(
            self.model.interaction_network.neighbors(self.unique_id)
        )
        if len(my_neighbors_ids) > 0:
            other_player = self.model.schedule.agents[random.choice(my_neighbors_ids)]

            choices = [
                self.reason(player, played_game) for player in [self, other_player]
            ]

            # print(choices)
            payoff = played_game.return_payoffs(
                [x + 1 for x in choices]
            )  # the +1 part is super hacky, it's because choices currently start at 1.
            # print(payoff)
            self.gathered_utility += payoff[0]

            # Did the opponent do, what I would have done if I were a team-reasoner?
            # This could be tied to the TR-function!
            # opponent_team_reasoned = played_game.tr_choice == choices[1]

            #         print(opponent_team_reasoned)

            #         if opponent_team_reasoned:

            # HOW DO WE DECOUPLE THE LEARNING FUNCTION HERE? By using e-utils?
            self.my_learner.update(payoff[0])  # 1 - int(opponent_team_reasoned))

            #         self.probability_team_reasoning = (
            #             (self.probability_team_reasoning * self.model.this_step)
            #             + int(opponent_team_reasoned)
            #         ) / (self.model.this_step + 1)

            pass
        else:
            self.my_learner.update(np.random.rand())  # random update for isolates
            pass

    def reason(self, player, played_game):
        """Reasoning function, in which the players determine their played strategy, given the game
        and whether they are team-reasoners.
        Args:
            player (MyAgent): An agent of class MyAgent.
            game (game): A game implemented according to our game-syntax.
        Returns:
            list: A strategy to be used in nashpy.
        """
        prediction = self.my_learner.return_prediction()
        # player.my_learner.return_prediction(estimate="probabilistic")

        if player.team_reasoner == True:

            self.team_reasoning_counter += 1  # useless atm
            self.probability_team_reasoning = prediction  # useless atm
            tr_strategies = played_game.return_TR_strategy(prediction)

            return random_argmax(
                np.mean(np.array(tr_strategies), axis=0)
            )  # in case of multiple different equilibria eg hilo
        else:
            if player == self:
                self.probability_team_reasoning = prediction
            return played_game.return_non_trs_choice(mode="random")


class team_reasoning_model(Model):
    """
    Attributes:
        datacollector (mesa datacollector): A mesa datacollector object ( defined internally)
        game (nashpy): a naspy-game which the actors play. (This might be later updated with the
            use of a dictionary, for multiple games.)
        n_agents (int): The number of agents.
        probability_team_reasoning (float or int): The probability that team-reasoning actors actually team reason. Between 0 and 1.
        proportion_team_reasoners (float or int): The share of team reasoners. Between 0 and 1. The actual share in the simulation
            will be rounded, depending on the number of agents in the simulation.
        schedule (mesa schedule): The stack of agents used internally by mesa.
        utility_calculation (str): How utilities for the replicator dynamic will be calculated. Either "empirical_average", using
            the actually collected payoffs from the last round, or "expected_utility", using the distribution of team-reasoners
            and the expected payoffs of the game(currently hardcoded!) as a base.
            The latter approach is faithful to Armadae and Lempert, 2015, and le.ads (qualitatively) to a smoother dynamic
    """

    def __init__(
        self,
        proportion_team_reasoners,
        n_agents,
        init_network,
        games,
        probability_team_reasoning,
        utility_calculation="empirical_average",
        learner=bayesian_beta_learner(
            prior={
                "type": "flat",
            }
        ),
        tr_threshold=0.5,
    ):
        """constructor class"""
        super().__init__()
        self.running = True  # for the batchrunner

        self.n_agents = n_agents
        self.schedule = MultiAgentActivation(self)
        self.games = games
        self.tr_threshold = tr_threshold
        self.probability_team_reasoning = probability_team_reasoning
        self.learner = learner

        self.make_n_agents(self.n_agents, proportion_team_reasoners)
        self.proportion_team_reasoners = proportion_team_reasoners
        self.utility_calculation = utility_calculation
        self.this_step = 0
        # print(self.games)
        # print(type(self.games))

        # This needs to be repaired:
        # if isinstance(self.games, game):  # Turn games into single-game mixtures:
        #     self.game_mixture = game_mixture([[self.games, 1.0]])
        # elif isinstance(self.games, game_mixture):
        #     self.game_mixture = self.games
        # else:
        #     raise Exception("games' must be of simple-teams game or game-mixture type.")
        self.game_mixture = self.games
        # what would team-reasoners and non-team-reasoners do? Calculate now, for speed!
        # This is bad:
        for this_game in self.game_mixture.games_list:
            this_game.tr_choice = this_game.return_team_reasoners_choice()
            this_game.non_tr_choice = this_game.return_non_trs_choice()
        print(self.game_mixture.games_list[0].tr_choice)
        # self.team_reasoners_choice = self.game.return_team_reasoners_choice()
        # self.non_trs_choice = self.game.return_non_trs_choice()

        self.game_generator = partial(self.game_mixture.return_game)

        # Initialize the networks:
        self.init_network = init_network
        if isinstance(init_network, nx.classes.graph.Graph):
            if len(list(init_network.nodes)) != self.n_agents:
                raise Exception(
                    "The number of nodes in the input-graph doesn't match up with the passed n_agents."
                )
            else:
                self.interaction_network = init_network
                self.is_changing_network = False

        elif isinstance(init_network, list) or isinstance(init_network, tuple):
            print("using a list of graphs")
            self.interaction_network = init_network[0]
            self.is_changing_network = True

        agent_reporters = {
            "team_reasoner": lambda a: getattr(a, "team_reasoner", None),
            "probability_team_reasoning": lambda a: getattr(
                a, "probability_team_reasoning", None
            ),
            "gathered_utility": lambda a: getattr(a, "gathered_utility", None),
            "my_learner": lambda a: getattr(a, "my_learner", None),
            "alive": lambda a: getattr(a, "alive", None),
            "team_reasoning_counter": lambda a: getattr(
                a, "team_reasoning_counter", None
            ),
        }

        model_reporters = {
            "proportion_team_reasoners": lambda a: getattr(
                a, "proportion_team_reasoners", None
            ),
        }

        self.datacollector = DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

    def make_n_agents(self, n_agents, proportion_team_reasoners, starting_no=0):
        """A helper function to set up a number of agents. Used at init and every step of the model.

        Args:
            n_agents (int): Number of agents.
            proportion_team_reasoners (float or int): The share of team reasoners. Between 0 and 1. The actual share in the simulation
                will be rounded, depending on the number of agents in the simulation.
            starting_no (int, optional): Number at which to start the unique id's of the agents.
                Necessary to adjust upwards when adding agents, so that earlier agents are not overwritten.
        """
        # team_reason_until = int(proportion_team_reasoners * n_agents)
        for ix, i in enumerate(range(self.n_agents)):
            if np.random.rand() <= proportion_team_reasoners:
                a = MyAgent(
                    unique_id=i + starting_no,
                    model=self,
                    team_reasoner=True,
                    probability_team_reasoning=self.probability_team_reasoning,
                )
            else:
                a = MyAgent(
                    unique_id=i + starting_no,
                    model=self,
                    team_reasoner=False,
                    probability_team_reasoning=self.probability_team_reasoning,
                )

            self.schedule.add(a)

    def step(self):
        self.this_step += 1
        if self.is_changing_network:
            self.interaction_network = self.init_network[self.this_step]
        self.schedule.step()
        self.datacollector.collect(self)
