You are an AI expert in artificial life especially cellular automata. Return ONLY the raw, valid Python code that represent a python function. This function will be part of cellular automata code that is used to generate a level for #game_name# game. The levels should be #game_description#.

The cellular automata will be applied for maximum of #steps# steps. If the system reach a solution before hand, it will stop and return the solution. In each step, a total of #local_functions# local functions (applied to #size#x#size# area at each cell location and returns local computation) and #global_functions# global functions (applied to the #width#x#height# map and returns a global computation over the whole map) is applied. All these functions are applied to each tile type (total of #values#) seperately.

The fitness of the cellular automata program is calculated based on the highest reached fitness value during the steps of the cellular automata and how fast it reached it only if it solved the problem.

Do NOT include markdown blocks like ```python, do NOT include comments, and do NOT write any introductory or concluding text.