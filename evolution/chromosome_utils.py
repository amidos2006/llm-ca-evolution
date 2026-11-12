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

_worker_envs = {}

def worker_env(config):
    # One environment per process, built on first use because it cannot be pickled.
    name = config['environment']['name']
    if name not in _worker_envs:
        _worker_envs[name] = pcg_benchmark.make(name)
    return _worker_envs[name]

def evaluate_chromosome(chromosome_class, payload):
    # Entry point for a worker process: rebuild the chromosome from code and return its fitness.
    # The payload holds the constructor arguments after the client, starting with the config.
    chromosome = chromosome_class(None, *payload)
    return chromosome.fitness(worker_env(payload[0]))
