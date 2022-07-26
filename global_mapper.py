from map_generator import heuristic_generator

class Node:
    def __init__(self, parent, position) -> None:
        self.parent = parent
        self.pos = position

    def __eq__(self, node):
        if node == None:
            return self.pos == node

        return self.pos == node.pos

    def __gt__(self, node):
        return self.pos > node.pos
    
    def __lt__(self, node):
        return self.pos < node.pos
    
def a_star(grid, init, goal, cost, delta, heuristic):
    """
    :param grid: 2D matrix of the world, with the obstacles marked as '1', rest as '0'
    :param init: list of x and y co-ordinates of the robot's initial position
    :param goal: list of x and y co-ordinates of the robot's intended final position
    :param cost: cost of moving one position in the grid
    :param delta: list of all the possible movements
    :param heuristic: 2D matrix of same size as grid, giving the cost of reaching the goal from each cell
    :return: path: list of the cost of the minimum path, and the goal position
    :return: extended: 2D matrix of same size as grid, for each element, the count when it was expanded or -1 if
             the element was never expanded.
    """
    path = []
    routes = []
    val = 1
    visited = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    visited[init[0]][init[1]] = 1

    expand = [[-1 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    expand[init[0]][init[1]] = 0

    init = Node(None, init)
    g = 0
    f = g + heuristic[init.pos[0]][init.pos[1]]

    minList = [f, g, init]

    while [minList[2].pos[0], minList[2].pos[1]] != goal:
        for i in range(len(delta)):
            point = Node(init, [init.pos[0] + delta[i][0], init.pos[1] + delta[i][1]])
            
            if 0 <= point.pos[0] < len(grid) and 0 <= point.pos[1] < len(grid[0]):
                if visited[point.pos[0]][point.pos[1]] == 0 and grid[point.pos[0]][point.pos[1]] == 0:
                    g2 = g + cost
                    f2 = g2 + heuristic[point.pos[0]][point.pos[1]]
                    path.append([f2, g2, point])
                    visited[point.pos[0]][point.pos[1]] = 1

        if not path:
            return 'fail', expand

        
        del minList[:]
        minList = min(path)
        routes.append(minList)
        path.remove(minList)
        init = minList[2]
        g = minList[1]
 
        expand[init.pos[0]][init.pos[1]] = val
        val += 1

    # print(routes)
    return routes, expand

def return_path(path):
    coord = []
    try:
        dest = path[-1][2]
        while True:
            if dest != None:
                coord.append(dest.pos)
                dest = dest.parent
            else:
                break

        return coord[::-1]
    except:
        return coord

def find_path(maze, start, end):

    h_map = heuristic_generator(maze, end)
    cost = 1

    delta = [[0, -1],  # go up
             [-1, 0],  # go left
             [0, 1],  # go down
             [1, 0]]  # go right

    path, expand = a_star(maze, start, end, cost, delta, h_map)
    return path, expand
