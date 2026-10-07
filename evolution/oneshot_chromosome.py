import numpy as np
import json
from .chromosome_utils import run_claude_code

class OneShotChromosome:
    def __init__(self, client, config, execution_function=None):
        self.client = client
        self.config = config
        self.fitness_value = None
        self.compiled = False

        with open(f"{self.config['llm']['system_prompt']}", "r") as f:
            self.system_prompt = f.read()
            self.system_prompt = self.system_prompt.replace("#game_name#", str(self.config['environment']['name']))
            self.system_prompt = self.system_prompt.replace("#game_description#", self.config['environment']['description'])
            self.system_prompt = self.system_prompt.replace("#width#", str(self.config['environment']['width']))
            self.system_prompt = self.system_prompt.replace("#height#", str(self.config['environment']['height']))
            self.system_prompt = self.system_prompt.replace("#values#", str(self.config['environment']['values']))

        with open(f"{self.config['llm']['user_execute_prompt']}", "r") as f:
            self.user_execute_prompt = f.read()

        with open(f"{self.config['llm']['user_crossover_prompt']}", "r") as f:
            self.user_crossover_prompt = f.read()

        self.execute_string = None
        if execution_function:
            self.execute_string = "" + execution_function
        else:
            self.execute_string = self.get_execute_function(True)

    def _fitness_meaning(self, value):
        if value <= -100:
            return "compile failed"
        if value < 0:
            return "runtime errors on some evaluation runs; more negative means more failures"
        if value < 1:
            return "level constraints are not fully satisfied; higher is better quality"
        return "constraints are satisfied; extra score is diversity across generated levels"

    def _fitness_text(self, value):
        if value is None:
            return ""
        return f"\nFITNESS: {value}\nMeaning: {self._fitness_meaning(value)}"

    def _task_instruction(self, initialization):
        if initialization:
            return "Write execute from scratch so it builds a complete level that satisfies the constraints."
        if not self.config['llm']['code_context']:
            return "Write a new execute from scratch. Keep the required name and signature."
        return (
            "Rewrite execute. Keep the name and signature. "
            "Produce a different implementation, not a paraphrase. "
            "If fitness is below 1, prioritize satisfying the level constraints. "
            "If fitness is 1 or higher, increase diversity without breaking solvability."
        )

    def _shape_replacements(self):
        return {
            "#width#": str(self.config['environment']['width']),
            "#height#": str(self.config['environment']['height']),
            "#values#": str(self.config['environment']['values']),
        }

    def get_execute_function(self, initialization):
        replacements = {
            **self._shape_replacements(),
            "#task_instruction#": self._task_instruction(initialization),
            "#current_code#": "",
            "#current_fitness#": "",
        }
        if self.config['llm']['code_context'] and not initialization:
            replacements["#current_code#"] = "\n\nFUNCTION TO EDIT:\n" + (self.execute_string or "")
            if self.fitness_value is not None:
                replacements["#current_fitness#"] = "\n" + self._fitness_text(self.fitness_value)
        return run_claude_code(self.client, self.config, self.system_prompt, self.user_execute_prompt, replacements)

    def get_crossover_function(self, other):
        # Fitness follows the same code_context switch as mutation.
        show_fitness = self.config['llm']['code_context']
        replacements = {
            **self._shape_replacements(),
            "#parent_a_code#": self.execute_string or "",
            "#parent_b_code#": other.execute_string or "",
            "#parent_a_fitness#": self._fitness_text(self.fitness_value) if show_fitness else "",
            "#parent_b_fitness#": self._fitness_text(other.fitness_value) if show_fitness else "",
        }
        return run_claude_code(self.client, self.config, self.system_prompt, self.user_crossover_prompt, replacements)

    def compile(self):
        self.namespace = {
            'np': np
        }

        total_errors = 0.0
        try:
            exec(self.execute_string, self.namespace)
        except:
            total_errors += 1.0
        self.execute_function = self.namespace.get('execute', None)

        self.compiled = True
        return total_errors

    def execute(self, env):
        # One call builds the whole level, so there are no steps to report.
        global_observation = np.array(env.content_space.sample(), dtype=np.int32)
        try:
            level = np.array(self.execute_function(global_observation.copy()), dtype=np.int32)
        except Exception as e:
            return 1.0, None, -1
        # The level goes straight to the evaluator, so reject anything it cannot read.
        if level.shape != global_observation.shape or level.min() < 0 or level.max() >= self.config['environment']['values']:
            return 1.0, None, -1
        return 0.0, level, 0.0

    def fitness(self, env):
        if self.fitness_value is None:
            compile_errors = self.compile()
            if compile_errors > 0.0:
                self.fitness_value = -100.0
            else:
                runtime_errors = 0
                levels = []
                env.content_space.seed(self.config['evolution']['seed'])
                for _ in range(self.config['evolution']['fitness_runs']):
                    errors, level, _ = self.execute(env)
                    if errors > 0.0:
                        runtime_errors += 1.0
                    else:
                        levels.append(level)
                if runtime_errors > 0.0:
                    self.fitness_value = -1 * runtime_errors / self.config['evolution']['fitness_runs']
                else:
                    q,d,_,details,_ = env.evaluate(levels)
                    if q >= 1.0:
                        self.fitness_value = 1.0 + (d + np.mean(details['diversity'])) / 2
                    else:
                        self.fitness_value = (q + np.mean(details['quality'])) / 2
        return self.fitness_value

    def mutate(self, rng):
        child = OneShotChromosome(self.client, self.config, self.execute_string)
        if rng.rand() < self.config['evolution']['mutation_rate']:
            child.execute_string = self.get_execute_function(False)
        return child

    def crossover(self, other, rng):
        child = OneShotChromosome(self.client, self.config, self.execute_string)
        if rng.rand() < self.config['evolution']['crossover_rate']:
            child.execute_string = self.get_crossover_function(other)
        return child

    def payload(self):
        # Everything a worker process needs to score this chromosome, all of it picklable.
        return (self.config, self.execute_string)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "execute_function": self.execute_string,
                "fitness": self.fitness_value,
            }, f)

    def save_image(self, env, filename):
        # Fitness may have run in a worker, so the code is not always compiled here.
        if not self.compiled:
            self.compile()
        errors, level, _ = self.execute(env)
        if errors == 0.0 and level is not None:
            try:
                env.render(level).save(filename)
            except:
                pass

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.execute_string = data.get("execute_function", "")
            self.fitness_value = data.get("fitness", None)

    def __str__(self):
        return self.execute_string or ""
