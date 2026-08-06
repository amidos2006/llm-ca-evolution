# Programmable Cellular Automata with LLM
The run take a config file to run stuff. The config file has 3 different things that I want to experiment.
- number of local functions (`local_functions`)
- number of global functions (`global_functions`)
- games (different files with different game `name` from the `pcg_benchmark`)
- having context for the llm to evolve like each mutation have the whole chromosome and fitness value as part of the prompt to help with diversity and having context (`code_context` can be True or False)

## To run
```python
uv run run.py -c configurations/binary.yaml
```