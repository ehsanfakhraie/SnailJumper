import copy

from player import Player
import numpy as np


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        self.pop_mode = 'rw'
        self.parent_mode = 'sus'
        self.log = []
        self.mutate_num = 0
        self.mutate_thresh = 0.2

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """

        sorted_players = sorted(players, key=lambda player: player.fitness, reverse=True)
        best_fitness = sorted_players[0].fitness
        worst_fitness = sorted_players[len(sorted_players) - 1].fitness
        fitnesses = [player.fitness for player in players]
        mean_fitness = sum(fitnesses) / len(fitnesses)

        fits = list(map(lambda player: player.fitness, players))
        max = np.max(fits)
        min = np.min(fits)
        average = np.average(fits)
        print([min, max, average, self.mutate_num, self.mutate_thresh])
        self.log.append((min, max, average, self.mutate_num))

        if self.pop_mode == 'rw':
            return self.roulette_wheel(players, num_players)
        elif self.pop_mode == 'sus':
            return self.sus_selector(players, num_players)
        else:
            sorted_players = sorted(players, key=lambda player: player.fitness, reverse=True)
            return sorted_players[: num_players]

    def roulette_wheel(self, players, num_players):
        probas = []
        sum_fitness = 0
        for player in players:
            sum_fitness += player.fitness
        for player in players:
            probas.append(player.fitness / sum_fitness)
        for i in range(1, len(players)):
            probas[i] += probas[i - 1]

        results = []
        randoms = []

        for j in range(num_players):
            random_number = np.random.uniform(0, 1, 1)
            for i, proba in enumerate(probas):
                if random_number <= proba:
                    results.append(self.clone_player(players[i]))
                    break

        return results

    def sus_selector(self, players, num_players):
        probas = []
        sum_fitness = 0
        for player in players:
            sum_fitness += player.fitness
        for player in players:
            probas.append(player.fitness / sum_fitness)
        for i in range(1, len(players)):
            probas[i] += probas[i - 1]

        random_number = np.random.uniform(0, 1 / num_players, 1)
        step = (probas[len(probas) - 1] - random_number) / num_players
        results = []

        for i in range(num_players):
            now = (i + 1) * step
            for i, proba in enumerate(probas):
                if now <= proba:
                    results.append(self.clone_player(players[i]))
                    break
        return results

    def add_noise(self, array, threshold):
        random_number = np.random.uniform(0, 1, 1)
        if random_number < threshold:
            array += np.random.randn(array.shape[0] * array.shape[1]).reshape(array.shape[0], array.shape[1])

    def mutate(self, child):
        threshold = 0.3
        self.add_noise(child.nn.layers[0], threshold)
        self.add_noise(child.nn.layers[1], threshold)
        self.add_noise(child.nn.biases[0], threshold)
        self.add_noise(child.nn.biases[1], threshold)

    def crossover(self, clayers1, clayers2, players1, players2):
        for i in range(len(players1)):
            layer1 = players1[i]
            layer2 = players2[i]
            length = len(layer1) * len(layer1[0])
            index = round(np.random.random() * length)
            layer3 = np.array(list(layer1.reshape(length)[:index]) + list(layer2.reshape(length)[index:])).reshape(
                [len(layer1), len(layer1[0])])
            layer4 = np.array(list(layer2.reshape(length)[:index]) + list(layer1.reshape(length)[index:])).reshape(
                [len(layer1), len(layer1[0])])
            clayers1[i] = layer3
            clayers2[i] = layer4

    def child_production(self, parent1, parent2):
        child1 = self.clone_player(parent1)
        child2 = self.clone_player(parent2)

        self.crossover(child1.nn.layers, child2.nn.layers, parent1.nn.layers, parent2.nn.layers)
        self.crossover(child1.nn.layers, child2.nn.layers, parent1.nn.layers, parent2.nn.layers)
        self.crossover(child1.nn.biases, child2.nn.biases, parent1.nn.biases, parent2.nn.biases)
        self.crossover(child1.nn.biases, child2.nn.biases, parent1.nn.biases, parent2.nn.biases)

        self.mutate(child1)
        self.mutate(child2)
        return child1, child2

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:

            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            prev_parents = prev_players.copy()
            children = []

            if self.parent_mode == 'rw':
                prev_parents = self.roulette_wheel(prev_parents, len(prev_parents))
            elif self.parent_mode == 'sus':
                prev_parents = self.sus_selector(prev_parents, len(prev_parents))

            for i in range(0, len(prev_parents), 2):
                child1, child2 = self.child_production(prev_parents[i], prev_parents[i + 1])
                children.append(child1)
                children.append(child2)

            return children

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
