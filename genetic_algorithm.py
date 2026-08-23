import anthropic
import pcg_benchmark
from ca_chromosome import CAChromosome, evaluate_chromosome
from tqdm import trange
import numpy as np 
import os
import parallel
import secrets

SEED_LIMIT = 2 ** 31 - 1

def breed(recipe):
    # Runs on a worker thread, so it gets its own generator instead of the global one.
    parent1, parent2, seed = recipe
    rng = np.random.RandomState(seed)
    child = parent1.crossover(parent2, rng)
    return child.mutate(rng)

class GeneticAlgorithm:
    def __init__(self, config):
        self.config = config
        np.random.seed(int(self.config['evolution']['seed']))
        parallel.configure(
            self.config['llm'].get('max_concurrency', parallel.DEFAULT_LIMIT),
            self.config['evolution'].get('fitness_workers', parallel.DEFAULT_PROCESSES),
        )

        self.claude_client = anthropic.Client()
        self.env = pcg_benchmark.make(self.config['environment']['name'])

        # Every chromosome is an independent batch of API calls.
        self.population = parallel.run_parallel(
            lambda _: CAChromosome(self.claude_client, self.config),
            range(self.config['evolution']['population']),
        )
        self.evaluate_population()
        self.population = sorted(self.population, key=lambda x: x.fitness_value, reverse=True)

    def evaluate_population(self):
        # Scoring a chromosome is pure CPU work, so hand the unscored ones to worker processes.
        pending = [chromosome for chromosome in self.population if chromosome.fitness_value is None]
        values = parallel.run_processes(evaluate_chromosome, [chromosome.payload() for chromosome in pending])
        for chromosome, value in zip(pending, values):
            chromosome.fitness_value = value

    def tournament(self):
        selected = []
        for i in range(self.config['evolution']['tournament']):
            idx = np.random.randint(0, len(self.population))
            selected.append(self.population[idx])
        sorted_selected = sorted(selected, key=lambda x: x.fitness_value, reverse=True)
        return sorted_selected[0]

    def best(self):
        self.evaluate_population()
        sorted_population = sorted(self.population, key=lambda x: x.fitness_value, reverse=True)
        return sorted_population[0]

    def step(self):
        new_population = []
        elites = max(1, int(self.config['evolution']['population'] * self.config['evolution']['elitism']))
        for i in range(elites):
            new_population.append(self.population[i])
        # Pick parents and child seeds on this thread so the run stays reproducible,
        # then breed the whole generation in parallel.
        recipes = []
        while len(new_population) + len(recipes) < self.config['evolution']['population']:
            parent1 = self.tournament()
            parent2 = self.tournament()
            recipes.append((parent1, parent2, int(np.random.randint(0, SEED_LIMIT))))
        new_population.extend(parallel.run_parallel(breed, recipes))
        self.population = new_population
        self.evaluate_population()
        self.population = sorted(self.population, key=lambda x: x.fitness_value, reverse=True)

    def fitness_stats(self):
        values = np.array([chromosome.fitness_value for chromosome in self.population], dtype=float)
        return {
            "best_fitness": float(np.max(values)),
            "mean_fitness": float(np.mean(values)),
            "median_fitness": float(np.median(values)),
            "min_fitness": float(np.min(values)),
            "std_fitness": float(np.std(values)),
        }

    def should_save(self, generation):
        # Snapshots are the bulk of the disk a sweep uses, so keep every save_every one plus the last.
        save_every = max(1, int(self.config['evolution'].get('save_every', 1)))
        return generation % save_every == 0 or generation == self.config['evolution']['generations']

    def run(self, on_generation=None):
        unique_id = secrets.token_hex(6)
        progress = trange(self.config['evolution']['generations'])
        best_chromosome = None
        self.save_to_folder(os.path.join(self.config['evolution']['save_folder'] + "_" + unique_id, f"generation_0"))
        if on_generation is not None:
            on_generation(0, self)
        for generation in progress:
            self.step()
            if self.should_save(generation + 1):
                self.save_to_folder(os.path.join(self.config['evolution']['save_folder'] + "_" + unique_id, f"generation_{generation+1}"))
            best_chromosome = self.best()
            progress.set_description(f"Best fitness: {best_chromosome.fitness_value:.4f}")
            if on_generation is not None:
                on_generation(generation + 1, self)
        return best_chromosome

    def save_to_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for index, chromosome in enumerate(self.population):
            chromosome.save_to_file(os.path.join(folder_path, f"chromosome_{index}.json"))
        if self.config['evolution']['save_images']:
            self.population[0].save_image(self.env, os.path.join(folder_path, "chromosome_0.png"))