You are an expert in cellular automata for procedural level generation. Return ONLY raw, valid Python source for a single function. No markdown fences, no imports, no print statements, no extra text. Comments are allowed.

This function is the update rule of a cellular automaton that generates a level for the game `#game_name#`. Target levels: #game_description#.

## How the automaton runs
The map starts as a random grid of `#height#` rows by `#width#` columns. Each cell is an integer tile id in `{0, ..., #values#-1}`. The automaton runs at most `#steps#` steps and stops early if the level is already solved.

Each step, `execute` is called once for every cell and returns the next tile id for that cell. #update_order#

## Function contract
- `def execute(local_observation, global_observation):`
- `local_observation` is a `#size#`x`#size#` numpy array of tile ids centered on the cell; the cell itself is `local_observation[#center#, #center#]`. Outside the map, the array holds border tiles.
- `global_observation` is a `#height#`x`#width#` numpy array of tile ids for the whole map.
- Return a single tile id in `{0, ..., #values#-1}`.

You may use `numpy` as `np` and native Python. The first line of your output must be the `def` line.

## Fitness scale
- `-100`: the program failed to compile
- `(-1, 0)`: runtime errors (more negative = more failing runs)
- `[0, 1)`: the level constraints are not fully satisfied (higher is better quality)
- `>= 1`: constraints are satisfied; the extra amount is diversity across generated levels
