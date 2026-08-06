import anthropic
import pcg_benchmark
from ca_chromosome import CAChromosome
from tqdm import trange
import numpy as np 
import os
import uuid

class GeneticAlgorithm:
    def __init__(self, config):
        self.config = config

        self.claude_client = anthropic.Client()
        self.env = pcg_benchmark.make(self.config['environment']['name'])

        population = []
        for _ in range(self.config['evolution']['population']):
            population.append(CAChromosome(self.claude_client, self.config))
        self.population = population
        for chromosome in self.population:
            chromosome.fitness(self.env)
        self.population = sorted(self.population, key=lambda x: x.fitness_value, reverse=True)

    def tournament(self):
        selected = []
        for i in range(self.config['evolution']['tournament']):
            idx = np.random.randint(0, len(self.population))
            selected.append(self.population[idx])
        sorted_selected = sorted(selected, key=lambda x: x.fitness_value, reverse=True)
        return sorted_selected[0]

    def best(self):
        for chromosome in self.population:
            chromosome.fitness(self.env)
        sorted_population = sorted(self.population, key=lambda x: x.fitness_value, reverse=True)
        return sorted_population[0]

    def step(self):
        new_population = []
        elites = max(1, int(self.config['evolution']['population'] * self.config['evolution']['elitism']))
        for i in range(elites):
            new_population.append(self.population[i])
        while len(new_population) < self.config['evolution']['population']:
            parent1 = self.tournament()
            parent2 = self.tournament()
            child = parent1.crossover(parent2)
            child = child.mutate()
            new_population.append(child)
        self.population = new_population
        for chromosome in self.population:
            chromosome.fitness(self.env)
        self.population = sorted(self.population, key=lambda x: x.fitness_value, reverse=True)

    def run(self):
        unique_id = str(uuid.uuid4())
        progress = trange(self.config['evolution']['generations'])
        best_chromosome = None
        self.save_to_folder(os.path.join(self.config['evolution']['save_folder'] + "_" + unique_id, f"generation_0"))
        for generation in progress:
            self.step()
            self.save_to_folder(os.path.join(self.config['evolution']['save_folder'] + "_" + unique_id, f"generation_{generation+1}"))
            best_chromosome = self.best()
            progress.set_description(f"Best fitness: {best_chromosome.fitness_value:.4f}")
        return best_chromosome

    def save_to_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for index, chromosome in enumerate(self.population):
            chromosome.save_to_file(os.path.join(folder_path, f"chromosome_{index}.json"))