import numpy as np
import json
from . import parallel
import pcg_benchmark
import time
from utils import usage

def run_claude_code(client, config, system_prompt, user_prompt, replacements):
    user_prompt = "" + user_prompt
    for key, value in replacements.items():
        user_prompt = user_prompt.replace(key, value)
    got_response = False
    attempt = 0
    while not got_response:
        try:
            # The slot keeps the whole process under the configured concurrency cap.
            with parallel.api_slot():
                response = client.messages.create(
                    model=config['llm']['model'],
                    max_tokens=config['llm']['max_tokens'],
                    output_config={"effort": config['llm'].get('effort', 'medium')},
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
            got_response = True
            usage.record(response)
        except Exception as e:
            attempt += 1
            delay = parallel.backoff_delay(attempt)
            print(f"Error in Claude API call: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

    # Models with extended thinking put a thinking block before the answer, so the
    # text is not always the first block.
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise RuntimeError(
            f"No text block in Claude response (stop_reason={response.stop_reason}, "
            f"blocks={[block.type for block in response.content]}). "
            "Thinking tokens share the max_tokens budget; raise llm.max_tokens."
        )
    if response.stop_reason == "max_tokens":
        print(
            f"Warning: response hit max_tokens ({config['llm']['max_tokens']}), "
            "the code is likely truncated. Raise llm.max_tokens."
        )
    return text

class CAChromosome:
    def __init__(self, client, config, local_functions=None, global_functions=None, execution_function=None):
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
            self.system_prompt = self.system_prompt.replace("#steps#", str(self.config['ca']['steps']))
            self.system_prompt = self.system_prompt.replace("#local_functions#", str(self.config['ca']['local_functions']))
            self.system_prompt = self.system_prompt.replace("#global_functions#", str(self.config['ca']['global_functions']))
            self.system_prompt = self.system_prompt.replace("#size#", str(self.config['ca']['size']))

        with open(f"{self.config['llm']['user_execute_prompt']}", "r") as f:
            self.user_execute_prompt = f.read()

        with open(f"{self.config['llm']['user_function_prompt']}", "r") as f:
            self.user_function_prompt = f.read()

        # The functions do not depend on each other, so ask for them all at once.
        # The empty list has to exist first because get_context reads it.
        if local_functions:
            self.local_string = list(local_functions)
        else:
            self.local_string = []
            self.local_string = parallel.run_parallel(
                lambda index: self.get_local_function(index, True),
                range(self.config['ca']['local_functions']),
            )

        if global_functions:
            self.global_string = list(global_functions)
        else:
            self.global_string = []
            self.global_string = parallel.run_parallel(
                lambda index: self.get_global_function(index, True),
                range(self.config['ca']['global_functions']),
            )

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

    def _task_instruction(self, initialization, kind):
        target = "this function" if kind == "function" else "execute"
        if initialization:
            if kind == "function":
                return "Write this feature detector from scratch. Compute something that could help execute decide the next tile (counts, flags, or structure)."
            return "Write execute from scratch so it can emit every needed tile type using the helper results above."
        if not self.config['llm']['code_context']:
            return f"Write a new {target} from scratch. Keep the required name and signature."
        return (
            f"Rewrite ONLY {target}. Keep the name and signature. "
            "Produce a different implementation, not a paraphrase. "
            "If fitness is below 1, prioritize satisfying the level constraints. "
            "If fitness is 1 or higher, increase diversity without breaking solvability."
        )

    def get_context(self, initialization, current_function):
        replacements = {
            "#code_context#": "",
            "#current_code#": "",
            "#current_fitness#": ""
        }

        if self.config['llm']['code_context'] and not initialization:
            replacements["#code_context#"] = "\n\nCURRENT PROGRAM:\n" + str(self)
            replacements["#current_code#"] = "\n\nFUNCTION TO EDIT:\n" + current_function
            if self.fitness_value is not None:
                replacements["#current_fitness#"] = (
                    f"\n\nFITNESS: {self.fitness_value}\n"
                    f"Meaning: {self._fitness_meaning(self.fitness_value)}"
                )

        return replacements

    def get_local_function(self, index, initialization):
        replacements = {
            "#function_name#": f"local_{index}",
            "#function_type#": "local",
            "#width#": str(self.config['ca']['size']),
            "#height#": str(self.config['ca']['size']),
            "#function_representation#": "the binary neighborhood around one cell for a single tile type",
            "#task_instruction#": self._task_instruction(initialization, "function"),
        }
        context_replacements = self.get_context(initialization, self.local_string[index] if index < len(self.local_string) else "")
        return run_claude_code(self.client, self.config, self.system_prompt, self.user_function_prompt, {**replacements, **context_replacements})

    def get_global_function(self, index, initialization):
        replacements = {
            "#function_name#": f"global_{index}",
            "#function_type#": "global",
            "#width#": str(self.config['environment']['width']),
            "#height#": str(self.config['environment']['height']),
            "#function_representation#": "a binary mask of one tile type over the whole map",
            "#task_instruction#": self._task_instruction(initialization, "function"),
        }
        context_replacements = self.get_context(initialization, self.global_string[index] if index < len(self.global_string) else "")
        return run_claude_code(self.client, self.config, self.system_prompt, self.user_function_prompt, {**replacements, **context_replacements})

    def get_execute_function(self, initialization):
        n_local = self.config['ca']['local_functions']
        n_global = self.config['ca']['global_functions']
        replacements = {
            "#parameter_names#": ", ".join([f"local_{i}" for i in range(n_local)] + [f"global_{i}" for i in range(n_global)]),
            "#total_functions#": str(n_local + n_global),
            "#n_local#": str(n_local),
            "#n_global#": str(n_global),
            "#values#": str(self.config['environment']['values']),
            "#local_function_code#": "\n\n".join(self.local_string) if self.local_string else "(none)",
            "#global_function_code#": "\n\n".join(self.global_string) if self.global_string else "(none)",
            "#task_instruction#": self._task_instruction(initialization, "execute"),
        }
        context_replacements = self.get_context(initialization, self.execute_string if self.execute_string else "")
        return run_claude_code(self.client, self.config, self.system_prompt, self.user_execute_prompt, {**replacements, **context_replacements})

    def compile(self):
        total_errors = 0.0
        max_errors = self.config['ca']['local_functions'] + self.config['ca']['global_functions'] + 1

        self.namespace = {
            'np': np
        }
        
        for func in self.local_string:
            try:
                exec(func, self.namespace)
            except:
                total_errors += 1.0
        self.local_functions = []
        for i in range(self.config['ca']['local_functions']):
            self.local_functions.append(self.namespace.get(f'local_{i}', None))

        for func in self.global_string:
            try:
                exec(func, self.namespace)
            except:
                total_errors += 1.0
        self.global_functions = []
        for i in range(self.config['ca']['global_functions']):
            self.global_functions.append(self.namespace.get(f'global_{i}', None))

        try:
            exec(self.execute_string, self.namespace)
        except:
            total_errors += 1.0
        self.execute_function = self.namespace.get('execute', None)

        self.compiled = True
        return total_errors / (1.0 * max_errors)

    def execute(self, env):
        shift = self.config['ca']['size'] // 2
        current_state = np.array(env.content_space.sample(), dtype=np.int32)
        for step in range(self.config['ca']['steps']):
            padded_state = np.pad(
                current_state,
                pad_width=shift,  # 1 element on all 4 sides
                mode='constant',
                constant_values=((self.config['environment']['border']['top'], self.config['environment']['border']['bottom']),
                                 (self.config['environment']['border']['left'], self.config['environment']['border']['right']))
            )
            new_state = np.zeros_like(current_state)
            hybrid_state = current_state.copy()
            if self.config['ca'].get('global_function_synchronous', True):
                global_parameters = [[] for _ in range(self.config['ca']['global_functions'])]
                for tile in range(self.config['environment']['values']):
                    index = 0
                    global_observation = (hybrid_state == tile).astype(int)
                    for func in self.global_functions:
                        try:
                            global_parameters[index].append(func(global_observation[:,:]))
                        except Exception as e:
                            return 1.0, None, -1
                        index += 1
            for y in range(shift, self.config['environment']['height'] + shift):
                for x in range(shift, self.config['environment']['width'] + shift):
                    if not self.config['ca'].get('global_function_synchronous', True):
                        global_parameters = [[] for _ in range(self.config['ca']['global_functions'])]
                        for tile in range(self.config['environment']['values']):
                            index = 0
                            global_observation = (hybrid_state == tile).astype(int)
                            for func in self.global_functions:
                                try:
                                    global_parameters[index].append(func(global_observation[:,:]))
                                except Exception as e:
                                    return 1.0, None, -1
                                index += 1
                    
                    local_parameters = [[] for _ in range(self.config['ca']['local_functions'])]
                    for tile in range(self.config['environment']['values']):
                        local_observation = padded_state[y - shift:y + shift + 1, x - shift:x + shift + 1]
                        local_observation = (local_observation == tile).astype(int)
                        index = 0
                        for func in self.local_functions:
                            try:
                                local_parameters[index].append(func(local_observation[:,:]))
                            except Exception as e:
                                return 1.0, None, -1
                            index += 1
                    parameters = local_parameters + global_parameters
                    try:
                        new_state[y - shift, x - shift] = self.execute_function(*parameters)
                        hybrid_state[y - shift, x - shift] = new_state[y - shift, x - shift]
                    except Exception as e:
                        return 1.0, None, -1
                        
            current_state = new_state
            q,_,_,_,_ = env.evaluate(current_state)
            if q >= 1.0:
                break
        return 0.0, current_state, step / self.config['ca']['steps']

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
        child = CAChromosome(self.client, self.config, self.local_string, self.global_string, self.execute_string)
        # Draw every decision first, then rewrite the picked functions in parallel.
        local_indices = [i for i in range(self.config['ca']['local_functions'])
                         if rng.rand() < self.config['evolution']['mutation_rate']]
        global_indices = [i for i in range(self.config['ca']['global_functions'])
                          if rng.rand() < self.config['evolution']['mutation_rate']]
        mutate_execute = rng.rand() < self.config['evolution']['mutation_rate']

        local_code = parallel.run_parallel(lambda index: self.get_local_function(index, False), local_indices)
        for index, code in zip(local_indices, local_code):
            child.local_string[index] = code
        global_code = parallel.run_parallel(lambda index: self.get_global_function(index, False), global_indices)
        for index, code in zip(global_indices, global_code):
            child.global_string[index] = code
        if mutate_execute:
            child.execute_string = self.get_execute_function(False)
        return child

    def crossover(self, other, rng):
        child = CAChromosome(self.client, self.config, self.local_string, self.global_string, self.execute_string)
        for i in range(self.config['ca']['local_functions']):
            if rng.rand() < self.config['evolution']['crossover_rate']:
                child.local_string[i] = other.local_string[i]
        for i in range(self.config['ca']['global_functions']):
            if rng.rand() < self.config['evolution']['crossover_rate']:
                child.global_string[i] = other.global_string[i]
        if rng.rand() < self.config['evolution']['crossover_rate']:
            child.execute_string = other.execute_string
        return child

    def payload(self):
        # Everything a worker process needs to score this chromosome, all of it picklable.
        return (self.config, self.local_string, self.global_string, self.execute_string)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "local_functions": self.local_string,
                "global_functions": self.global_string,
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
            self.local_string = data.get("local_functions", [])
            self.global_string = data.get("global_functions", [])
            self.execute_string = data.get("execute_function", "")
            self.fitness_value = data.get("fitness", None)

    def __str__(self):
        string_values = ""
        if len(self.local_string) > 0:
            string_values += "\n\n".join(self.local_string)
        if len(self.global_string) > 0:
            string_values += "\n\n" + "\n\n".join(self.global_string)
        if self.execute_string:
            string_values += "\n\n" + self.execute_string
        return string_values

_worker_envs = {}

def worker_env(config):
    # One environment per process, built on first use because it cannot be pickled.
    name = config['environment']['name']
    if name not in _worker_envs:
        _worker_envs[name] = pcg_benchmark.make(name)
    return _worker_envs[name]

def evaluate_chromosome(payload):
    # Entry point for a worker process: rebuild the chromosome from code and return its fitness.
    config, local_string, global_string, execute_string = payload
    chromosome = CAChromosome(None, config, local_string, global_string, execute_string)
    return chromosome.fitness(worker_env(config))