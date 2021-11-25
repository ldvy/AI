from time import time
from bfa_search import breadth_first_search
from astar_search import astar_search
from puzzle import Puzzle

state = [[3, 2, 8,
          6, 4, 7,
          1, 0, 5],

         [3, 0, 1,
          7, 8, 2,
          5, 6, 4],

         [2, 3, 0,
          1, 4, 5,
          7, 8, 6]]

for i in range(len(state)):
    print('----- Puzzle Num:', i + 1, '-----')

    Puzzle.num_of_instances = 0
    t0 = time()
    bfs = breadth_first_search(state[i])
    t1 = time() - t0
    print('BFS:', bfs)
    print('states:', Puzzle.num_of_instances)
    print('time:', t1)
    print()

    Puzzle.num_of_instances = 0
    t0 = time()
    astar = astar_search(state[i])
    t1 = time() - t0
    print('A*:', astar)
    print('states:', Puzzle.num_of_instances)
    print('time:', t1)
    print('\n')
