import sys

from maze import Maze, path_from


def find_length(node):
    g = len(path_from(node))
    nodeE = maze.find_node('E')
    return abs(nodeE.x - node.x) + abs(nodeE.y - node.y) + g


def dfs(maze):
    start_node = maze.find_node('S')
    s = [start_node]
    while len(s) > 0:
        node = s.pop()  # stack
        if node.type == 'E':
            return path_from(node)
        if not node.visited:
            node.visited = True
            children = maze.get_possible_movements(node)
            children.sort(key=find_length)
            for child in children:
                if not child.visited:
                    child.parent = node
                    s.append(child)

    return None


maze = Maze.from_file(sys.argv[1])
maze.draw()
maze.path = dfs(maze)
print()
maze.draw()
