import sys

from maze import Maze, path_from


def bfs(maze):
    start_node = maze.find_node('S')
    q = [start_node]
    start_node.visited = True
    while len(q) > 0:
        node = q.pop(0)  # FIFO
        children = maze.get_possible_movements(node)
        for child in children:
            if not child.visited:
                child.parent = node
                child.visited = True
                q.append(child)
                if child.type == 'E':
                    return path_from(child)
    return None


maze = Maze.from_file(sys.argv[1])
maze.draw()
maze.path = bfs(maze)
print()
maze.draw()
