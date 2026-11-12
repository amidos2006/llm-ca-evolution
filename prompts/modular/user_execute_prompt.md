Write a Python function named execute that chooses the next tile id for one cell.

The first line must be:
def execute(#parameter_names#):

`execute` is called once per cell, once per CA step. It takes #total_functions# parameters named: #parameter_names#.
The first #n_local# parameters are local-function results; the remaining #n_global# are global-function results.
Each parameter is a sequence of length #values#. Index `t` is that helper's integer result on the binary mask for tile type `t`.
Return a single integer tile id in `{0, ..., #values#-1}`. Index arguments like `local_0[t]`.

Current helper functions (they already exist; use their results, do not redefine them):

LOCAL:
#local_function_code#

GLOBAL:
#global_function_code#

You may use numpy as np and native Python. Do not import anything. Do not print. Output only the function source. No markdown.

#task_instruction#
#code_context#
#current_code#
#current_fitness#
