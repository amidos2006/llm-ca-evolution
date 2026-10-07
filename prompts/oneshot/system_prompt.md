You are an expert in procedural level generation. Return ONLY raw, valid Python source for a single function. No markdown fences, no imports, no print statements, no extra text. Comments are allowed.

This function generates a complete level for the game `#game_name#` in a single call. Target levels: #game_description#.

## How the generator runs
`execute` is called once per level. It receives a random grid of `#height#` rows by `#width#` columns, where each cell is an integer tile id in `{0, ..., #values#-1}`, and must return the finished level. Nothing runs after it: there are no further steps or repair passes. Every call gets a different random grid, so use it as a source of variation between levels.

## Function contract
- `def execute(global_observation):`
- `global_observation` is a `#height#`x`#width#` numpy array of random tile ids.
- Return a `#height#`x`#width#` array (numpy array or nested lists) of integer tile ids in `{0, ..., #values#-1}`. Any other shape or value counts as a runtime error.

You may use `numpy` as `np` and native Python. The first line of your output must be the `def` line.

## Fitness scale
- `-100`: the program failed to compile
- `(-1, 0)`: runtime errors (more negative = more failing runs)
- `[0, 1)`: the level constraints are not fully satisfied (higher is better quality)
- `>= 1`: constraints are satisfied; the extra amount is diversity across generated levels
