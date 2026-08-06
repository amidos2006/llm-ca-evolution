import numpy as np
import json

def run_claude_code(client, config, system_prompt, user_prompt, replacements):
    user_prompt = "" + user_prompt
    for key, value in replacements.items():
        user_prompt = user_prompt.replace(key, value)
    got_response = False
    while not got_response:
        try:
            response = client.messages.create(
                model=config['llm']['model'],
                max_tokens=config['llm']['max_tokens'],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            got_response = True
        except Exception as e:
            print(f"Error in Claude API call: {e}. Retrying...")
    return response.content[0].text

class CAChromosome:
    def __init__(self, client, config, local_functions=None, global_functions=None, execution_function=None):
        self.client = client
        self.config = config
        self.fitness_value = None

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
            self.user_execute_prompt = self.user_execute_prompt.replace("#local_functions#", str(self.config['ca']['local_functions']))
            self.user_execute_prompt = self.user_execute_prompt.replace("#total_functions#", str(self.config['ca']['global_functions'] + self.config['ca']['local_functions']))
            self.user_execute_prompt = self.user_execute_prompt.replace("#values#", str(self.config['environment']['values']))

        with open(f"{self.config['llm']['user_function_prompt']}", "r") as f:
            self.user_function_prompt = f.read()

        if local_functions:
            self.local_string = list(local_functions)
        else:
            self.local_string = []
            for i in range(self.config['ca']['local_functions']):
                self.local_string.append(self.get_local_function(i, True))

        if global_functions:
            self.global_string = list(global_functions)
        else:
            self.global_string = []
            for i in range(self.config['ca']['global_functions']):
                self.global_string.append(self.get_global_function(i, True))

        self.execute_string = None
        if execution_function:
            self.execute_string = "" + execution_function
        else:
            self.execute_string = self.get_execute_function(True)

    def get_context(self, initialization, current_function):
        replacements = {
            "#code_context#": "",
            "#current_code#": "",
            "#current_fitness#": ""
        }

        if self.config['llm']['code_context'] and not initialization:
            replacements["#code_context#"] = "\n\nThe current cellular automata code is as follows:\n\n" + str(self)
            replacements["#current_code#"] = "\n\nThe current function that need to be modified is:\n\n" + current_function
            if self.fitness_value is not None:
                replacements["#current_fitness#"] = f"\n\nThe current fitness value of that cellular automata is {self.fitness_value}"
        
        return replacements


    def get_local_function(self, index, initialization):
        replacements = {
            "#function_name#": f"local_{index}",
            "#function_type#": "local",
            "#width#": str(self.config['ca']['size']),
            "#height#": str(self.config['ca']['size']),
            "#function_representation#": "the local observation around each cell location",
        }
        context_replacements = self.get_context(initialization, self.local_string[index] if index < len(self.local_string) else "")
        return run_claude_code(self.client, self.config, self.system_prompt, self.user_function_prompt, {**replacements, **context_replacements})

    def get_global_function(self, index, initialization):
        replacements = {
            "#function_name#": f"global_{index}",
            "#function_type#": "global",
            "#width#": str(self.config['environment']['width']),
            "#height#": str(self.config['environment']['height']),
        }
        context_replacements = self.get_context(initialization, self.global_string[index] if index < len(self.global_string) else "")
        return run_claude_code(self.client, self.config, self.system_prompt, self.user_function_prompt, {**replacements, **context_replacements})

    def get_execute_function(self, initialization):
        replacements = {
            "#global_functions#": "\n\n".join(self.global_string),
            "#local_functions#": "\n\n".join(self.local_string),
            "#parameter_names#": ", ".join([f"local_{i}" for i in range(self.config['ca']['local_functions'])] + [f"global_{i}" for i in range(self.config['ca']['global_functions'])]),
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
            global_parameters = [[]] * self.config['ca']['global_functions']
            for tile in range(self.config['environment']['values']):
                index = 0
                global_observation = (current_state == tile).astype(int)
                for func in self.global_functions:
                    try:
                        global_parameters[index].append(func(global_observation[:,:]))
                    except Exception as e:
                        return 1.0, None, -1
                    index += 1
            for y in range(shift, self.config['environment']['height'] + shift):
                for x in range(shift, self.config['environment']['width'] + shift):
                    local_parameters = [[]] * self.config['ca']['local_functions']
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
                        new_state[y-1, x-1] = self.execute_function(*parameters)
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

    def mutate(self):
        child = CAChromosome(self.client, self.config, self.local_string, self.global_string, self.execute_string)
        for i in range(self.config['ca']['local_functions']):
            if np.random.rand() < self.config['evolution']['mutation_rate']:
                child.local_string[i] = self.get_local_function(i, False)
        for i in range(self.config['ca']['global_functions']):
            if np.random.rand() < self.config['evolution']['mutation_rate']:
                child.global_string[i] = self.get_global_function(i, False)
        if np.random.rand() < self.config['evolution']['mutation_rate']:
            child.execute_string = self.get_execute_function(False)
        return child

    def crossover(self, other):
        child = CAChromosome(self.client, self.config, self.local_string, self.global_string, self.execute_string)
        for i in range(self.config['ca']['local_functions']):
            if np.random.rand() < self.config['evolution']['crossover_rate']:
                child.local_string[i] = other.local_string[i]
        for i in range(self.config['ca']['global_functions']):
            if np.random.rand() < self.config['evolution']['crossover_rate']:
                child.global_string[i] = other.global_string[i]
        if np.random.rand() < self.config['evolution']['crossover_rate']:
            child.execute_string = other.execute_string
        return child

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "local_functions": self.local_string,
                "global_functions": self.global_string,
                "execute_function": self.execute_string,
                "fitness": self.fitness_value,
            }, f)

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