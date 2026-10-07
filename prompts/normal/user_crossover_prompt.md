Combine two parent versions of the execute function into one child execute function.

The first line must be:
def execute(#parameter_names#):

`local_observation` is a #size#x#size# numpy array of tile ids centered on the cell.#global_description#
Return a single integer tile id in `{0, ..., #values#-1}`.

PARENT A:
#parent_a_code#
#parent_a_fitness#

PARENT B:
#parent_b_code#
#parent_b_fitness#

Write a child that mixes the rules of both parents into one working function. Do not return either parent unchanged. Keep the parts that help satisfy the level constraints, and favor the stronger parent where they conflict.

You may use numpy as np and native Python. Do not import anything. Do not print. Output only the function source. No markdown.
