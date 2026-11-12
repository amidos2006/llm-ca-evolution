You are an expert in cellular automata for procedural level generation. Return ONLY raw, valid Python source for a single function. No markdown fences, no imports, no print statements, no extra text. Comments are allowed.

This function is one gene in a programmable cellular automaton that generates a level for the game `#game_name#`. Target levels: #game_description#.

## How the automaton runs
The map starts as a random `#width#`x`#height#` grid. Each cell is an integer tile id in `{0, ..., #values#-1}`. The automaton runs at most `#steps#` steps and stops early if the level is already solved.

Each step, for every tile type `t` in `0 .. #values#-1`:
1. Build a binary mask `(grid == t)`.
2. Run each of the `#global_functions#` global functions on that full `#width#`x`#height#` mask. Each returns one integer.
3. For every cell, run each of the `#local_functions#` local functions on the `#size#`x`#size#` neighborhood mask of that cell (the map is padded at the borders). Each returns one integer.

Then, for every cell, `execute` receives one length-`#values#` vector per helper (index `t` is that helper's result for tile type `t`) and returns the next tile id for that cell.

## Function contracts
- Local: `def local_i(obs):` where `obs` is a `#size#`x`#size#` numpy array of 0s and 1s. Return one int (a feature, not a tile id).
- Global: `def global_i(obs):` where `obs` is a `#width#`x`#height#` numpy array of 0s and 1s. Return one int (a feature, not a tile id).
- Execute: `def execute(...):` takes one argument per helper, each a length-`#values#` sequence of ints. Return a single tile id in `{0, ..., #values#-1}`.

You may use `numpy` as `np` and native Python. The first line of your output must be the `def` line.

## Fitness scale
- `-100`: the program failed to compile
- `(-1, 0)`: runtime errors (more negative = more failing runs)
- `[0, 1)`: the level constraints are not fully satisfied (higher is better quality)
- `>= 1`: constraints are satisfied; the extra amount is diversity across generated levels
