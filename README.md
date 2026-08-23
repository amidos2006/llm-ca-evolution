# Programmable Cellular Automata with LLM
The run take a config file to run stuff. The config file has 3 different things that I want to experiment.
- number of local functions (`local_functions`)
- number of global functions (`global_functions`)
- games (different files with different game `name` from the `pcg_benchmark`)
- having context for the llm to evolve like each mutation have the whole chromosome and fitness value as part of the prompt to help with diversity and having context (`code_context` can be True or False)

## To run
You need ANTHROPIC API KEY before running

```python
uv run run.py -c configurations/binary.yaml
```

## To sweep
`sweep.yaml` holds the grid (values per key, plus the seeds). Check the size first, then run one slice or all of it.

```bash
uv run sweep.py --sweep-file sweep.yaml --count
uv run sweep.py --sweep-file sweep.yaml --shard 1/4
```

`run_sweep_tmux.sh` splits the grid into shards and gives each one a tmux window, so the shards run side by side.

```bash
./run_sweep_tmux.sh -n 4 -a
```

Inside a run, the API calls of a generation go out in parallel and the fitness of a generation is scored in worker processes. Three config keys control the load, and all of them are per shard, so 8 shards use 8 times what they say.
- `llm.max_concurrency`: API calls in flight at once. Lower it first if the shard logs fill with retries.
- `evolution.fitness_workers`: processes used for fitness. 1 keeps everything in one process, 0 takes every core.
- `evolution.save_every`: how often to write a population snapshot. Snapshots are what fills the disk.

The defaults are sized for the vast.ai box: 24 vCPUs, 48 GB RAM, 32 GB disk. That means 8 shards of 4 fitness workers and 6 concurrent API calls each. The GPU is never used, the work is CPU and API bound.