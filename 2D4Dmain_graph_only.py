import numpy as np
from tqdm import trange
import random
import os
import time
import cv2
import math
np.set_printoptions(threshold=np.inf)

""" ###################### VERTEX ####################"""

class Vertex:
    def __init__(self, node, mat):
        self.id = node
        self.mcv = mat
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_mcv(self):
        return self.mcv

    def set_mcv(self, tempMCV):
        self.mcv = tempMCV

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]


""" ###################### GRAPH ####################"""

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node, mat):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node, mat)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def get_vertex_mcv(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n].get_mcv()
        else:
            return None

    def set_vertex_mcv(self, n, tempMCV):
        self.vert_dict[n].set_mcv(tempMCV)

    def add_edge(self, frm, frm_mat, to, to_mat, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm, frm_mat)
        if to not in self.vert_dict:
            self.add_vertex(to, to_mat)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def printGraph(self):

        #"""
        for v in g:
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                #print('( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w)))
                print('( %s , %s, %s)'  % ( vid, wid, v.get_weight(w)))
        #"""

        """
        for v in g:
            print('g.vert_dict[%s]=%s' %(v.get_id(), g.vert_dict[v.get_id()]))
        """
        """
        for v in g:
            vid = v.get_id()
            print('( %s , \n%s)' % (vid, v.get_mcv()))
        """

    def plotGraph(self):

        imageMCV = np.zeros((realgrid,realgrid))

        imageExpanded = np.zeros((realgrid*7,realgrid*2*7))

        for v in g:

            vid = v.get_id()
            vid = vid.split(",")

            vidx = vid[0].split("(")
            vidy = vid[1].split(")")

            x = int(vidx[1])
            y = int(vidy[0])

            imageMCV[x][y] = 255

        imageGridFull = np.concatenate([imageMCV, [[0 if item == 0 else 255 for item in row] for row in grid]], axis=1)

        for i in range(0, imageGridFull.shape[0]):
            for ii in range(0, 7):
                for j in range(0, imageGridFull.shape[1]):
                    for jj in range(0, 7):
                        imageExpanded[i * 7 + ii][j * 7 + jj] = imageGridFull[i][j]

        cv2.imwrite("2d/curiosity_map.png", imageExpanded)


    def plotTrajectory(self, trajectory, t):

        imageMCV = np.asarray([[0 if item == 0 else 255 for item in row] for row in grid])
        imageExpanded = np.zeros((realgrid*7,realgrid*7))


        for v in trajectory:

            vid = v.split(",")

            vidx = vid[0].split("(")
            vidy = vid[1].split(")")

            x = int(vidx[1])
            y = int(vidy[0])

            imageMCV[x][y] = 100

        imageMCV[agent_x][agent_y] = 200
        imageMCV[goal_x][goal_y] = 50

        for i in range(0, imageMCV.shape[0]):
            for ii in range(0, 7):
                for j in range(0, imageMCV.shape[1]):
                    for jj in range(0, 7):
                        imageExpanded[i * 7 + ii][j * 7 + jj] = imageMCV[i][j]

        cv2.imwrite("2d/trajectory_map_"+str(t)+".png", imageExpanded)


    def plotGraphBuilding(self, t):

        imageGrid = np.asarray([[0 if item == 0 else 255 for item in row] for row in grid])
        imagemotion = np.asarray([[0 if item == 0 else 255 for item in row] for row in grid])

        imageMCV = np.zeros((realgrid,realgrid))
        imageFog = np.zeros((realgrid, realgrid))

        imageExpanded = np.zeros((realgrid*2*7,realgrid*2*7))

        for v in g:

            vid = v.get_id()
            vid = vid.split(",")

            vidx = vid[0].split("(")
            vidy = vid[1].split(")")

            x = int(vidx[1])
            y = int(vidy[0])

            imageMCV[x][y] = 100

            imageFog[x-2:x+3,y-2:y+3] = imageGrid[x-2:x+3,y-2:y+3]

        imagemotion[previous_x, previous_y] = 200
        imagemotion[agent_x, agent_y] = 100

        imageFull = np.concatenate([
            np.concatenate([imageMCV, imagemotion], axis=1),
            np.concatenate([[[0 if item == 0 else 255 for item in line] for line in fog], imageFog], axis=1)
        ], axis=0)

        for i in range(0, imageFull.shape[0]):
            for ii in range(0, 7):
                for j in range(0, imageFull.shape[1]):
                    for jj in range(0, 7):
                        imageExpanded[i * 7 + ii][j * 7 + jj] = imageFull[i][j]

        cv2.imwrite("2d/generated_map_"+str(t)+".png", imageExpanded)



def randomValidPoint(grid_type):

    if grid_type == "SmallGrid":
        x, y = 0, 0
        while smallGrid[x][y] == 0:
            x = random.randint(1, realgrid - 2)
            y = random.randint(1, realgrid - 2)
    elif grid_type == "FullGrid":
        x, y = 0, 0
        while grid[x][y] == 0:
            x = random.randint(1, realgrid - 2)
            y = random.randint(1, realgrid - 2)

    return x, y








g = Graph()

# initialize grid
"""
grid = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,1,0,0,1,1,1,1,0,0],
                [0,0,1,0,0,1,0,0,1,0,0,1,0,0],
                [0,0,1,0,0,1,0,0,1,0,0,1,0,0],
                [0,0,1,0,0,1,0,0,1,0,0,1,0,0],
                [0,0,1,0,0,1,0,0,1,0,0,1,0,0],
                [0,0,1,0,0,1,1,1,1,0,0,1,0,0],
                [0,0,1,0,0,1,0,0,1,0,0,1,0,0],
                [0,0,1,0,0,1,0,0,1,0,0,1,0,0],
                [0,0,1,0,0,1,0,0,1,0,0,1,0,0],
                [0,0,1,1,1,1,0,0,1,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
"""



grid = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,1,0,1,1,0,0,1,1,0,1,0,0],
                [0,0,1,0,1,1,0,0,1,1,0,1,0,0],
                [0,0,1,0,1,1,0,0,1,1,0,1,0,0],
                [0,0,1,0,1,1,0,0,1,1,0,1,0,0],
                [0,0,0,0,1,0,0,0,0,1,0,0,0,0],
                [0,0,1,0,1,1,0,0,1,1,0,1,0,0],
                [0,0,1,0,1,1,0,0,1,1,0,1,0,0],
                [0,0,1,0,1,1,0,0,1,1,0,1,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

"""
# initialize grid
smallGrid = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,1,0,0,1,1,1,1,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,1,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,1,0,0,1,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
                    
"""

smallGrid = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,1,1,1,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])



realgrid = grid.shape[0]

boundarymask = np.array([[1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1]])

"""
initMCV = np.array([[1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]])
                    
"""

initMCV = np.array([[255, 255, 255, 255, 255],
                    [255, 255, 255, 255, 255],
                    [255, 255, 255, 255, 255],
                    [255, 255, 255, 255, 255],
                    [255, 255, 255, 255, 255]])


agent_x, agent_y = 0, 0
move_x, move_y = 0, 0
goal_x, goal_y = 0, 0
previous_x, previous_y = 0, 0
next_x, next_y = 0, 0
bDirection = np.zeros((realgrid, realgrid))
fDirection = np.zeros((realgrid, realgrid))
reachability = np.zeros((5, 5))
traversal_status = {}
graph = {}
fog = np.zeros((realgrid, realgrid))

boundarySwitch = False
fDirectionSwitch = False
bDirectionSwitch = False

"""    
"""


def floodfill(x, y):

    global reachability

    #"hidden" stop clause - not reinvoking for "c" or "b", only for "a".
    if reachability[x][y] == 1:
        reachability[x][y] = 2
        #recursively invoke flood fill on all surrounding cells:
        if x > 0:
            floodfill(x-1,y)
        if x < len(reachability[y]) - 1:
            floodfill(x+1,y)
        if y > 0:
            floodfill(x,y-1)
        if y < len(reachability) - 1:
            floodfill(x,y+1)




def initializeMCV2D(point_x, point_y):

    #only if it uncovers new reachable area

    global traversal_status
    global boundarySwitch
    global reachability

    # add this point as vertex
    vertex = "(" + str(point_x) + "," + str(point_y) + ")"

    print(point_x, point_y)

    # hardcoded reachability discovery from grid
    reachability = np.zeros((5, 5))

    tempGrid = np.empty_like(grid)
    tempGrid[:] = grid

    reachability = np.asarray(tempGrid[point_x-2:point_x+3, point_y-2:point_y+3])

    floodfill(2, 2)

    reachability = np.asarray([[1 if item == 2 else 0 for item in row] for row in reachability])

    temp = np.empty_like(fog)
    temp[:,:] = fog[:,:]

    for i in range(0,5):
        for j in range(0,5):
            if reachability[i,j] == 1:
                temp[point_x-2+i,point_y-2+j] = reachability[i,j]

    if np.array_equal(temp, fog):
        print("No new area uncovered")
    else:

        print(fog)
        print(temp)

        fog[:,:] = temp[:,:]

        # Initialize MCV
        tempMCV = initMCV * reachability
        # mask boundary
        if boundarySwitch:
            tempMCV = tempMCV * boundarymask
        # add vertex and set traversal status
        g.add_vertex(vertex, tempMCV)
        traversal_status[vertex] = "Remaining"


"""    
"""

def updateMCV2D():

    global previous_x
    global previous_y
    global agent_x
    global agent_y

    vertex_agent = "(" + str(agent_x) + "," + str(agent_y) + ")"
    vertex_previous = "(" + str(previous_x) + "," + str(previous_y) + ")"

    bMask = np.ones((realgrid,realgrid))
    fMask = np.ones((realgrid,realgrid))

    if previous_x <= agent_x:
        if previous_y <= agent_y:
            bMask[previous_x:agent_x+1, previous_y:agent_y+1] = 0.0
            fMask[previous_x:agent_x+1, previous_y:agent_y+1] = 0.0
        else:
            bMask[previous_x:agent_x+1, agent_y:previous_y+1] = 0.0
            fMask[previous_x:agent_x+1, agent_y:previous_y+1] = 0.0
    else:
        if previous_y <= agent_y:
            bMask[agent_x:previous_x+1, previous_y:agent_y+1] = 0.0
            fMask[agent_x:previous_x+1, previous_y:agent_y+1] = 0.0
        else:
            bMask[agent_x:previous_x+1, agent_y:previous_y+1] = 0.0
            fMask[agent_x:previous_x+1, agent_y:previous_y+1] = 0.0

    tempMCV_agent = g.get_vertex_mcv(vertex_agent)
    tempMCV_previous = g.get_vertex_mcv(vertex_previous)

    # enable below to lower curiosity from backward direction
    if bDirectionSwitch and type(tempMCV_agent) is np.ndarray:
        print("from agent updateMCV \n",tempMCV_agent)
        bDirection = bMask[agent_x-2:agent_x+3,agent_y-2:agent_y+3]
        tempMCV_agent = tempMCV_agent * bDirection
        g.set_vertex_mcv(vertex_agent, tempMCV_agent)
        unique = np.unique(tempMCV_agent, return_counts=True)
        if unique[0].shape[0] == 1:
            traversal_status[vertex_agent] = "Done"

    # enable below to raise curiosity in forward direction
    if fDirectionSwitch and type(tempMCV_previous) is np.ndarray:
        print("from previous updateMCV \n",tempMCV_previous)
        fDirection = fMask[previous_x-2:previous_x+3,previous_y-2:previous_y+3]
        tempMCV_previous = tempMCV_previous * fDirection
        g.set_vertex_mcv(vertex_previous, tempMCV_previous)
        unique = np.unique(tempMCV_previous, return_counts=True)
        if unique[0].shape[0] == 1:
            traversal_status[vertex_previous] = "Done"

    print("last checkpoint")
    return bMask, fMask

"""    
"""

def manipulateGraph2D():

    global previous_x
    global previous_y
    global agent_x
    global agent_y

    foundcurrent = False
    foundprevious = False

    if g.num_vertices > 0:
        # Non Empty Graph
        for v in g:
            vid = v.get_id()

            if vid == "("+str(previous_x)+","+str(previous_y)+")":
                foundprevious = True

            if vid == "("+str(agent_x)+","+str(agent_y)+")":
                foundcurrent = True

    if not foundcurrent:
        print("initialize agent")
        initializeMCV2D(agent_x, agent_y)
    if not foundprevious:
        print("initialize previous")
        initializeMCV2D(previous_x, previous_y)

    updateMCV2D()

def plan2D():

    global traversal_status
    global move_x
    global move_y
    global next_x
    global next_y

    findStatus = False

    for v in traversal_status:
        if traversal_status[v] == "Remaining":
            findStatus = findaction(v)
            if findStatus:
                vertex = g.get_vertex(v)
                vid = vertex.get_id()
                vid = vid.split(",")
                vidx = vid[0].split("(")
                vidy = vid[1].split(")")
                next_x = int(vidx[1])
                next_y = int(vidy[0])
                break
        if findStatus:
            break
    if not findStatus:
        print("finished building graph")
        return True
    return False



def findaction(vertex):

    global move_x
    global move_y

    tempMCV = g.get_vertex_mcv(vertex)
    unique = np.unique(tempMCV, return_counts=True)
    best = unique[0][-1]
    count = unique[1][-1]

    # check if there are multiple curiosities to mark nodes that have all its neighbours visited
    if unique[0].shape[0] > 1:
        # find if highest curiosity has multiple counts
        # if yes : action random
        # acts as a tiebreaker for initialized MCV and randomize exact move in the direction
        if count > 1:
            searchIndex = 0
            actionIndex = random.randint(1, count)
            for i in range(0, 5):
                for j in range(0, 5):
                    if tempMCV[i][j] == best:
                        searchIndex += 1
                    if searchIndex == actionIndex:
                        move_x, move_y = i, j
                        return True
        else:  # else  : max curiosity
            move_x, move_y = np.unravel_index(tempMCV.argmax(), tempMCV.shape)
            return True
    else:
        return False

"""    
"""

def buildGraph(max_steps):

    global previous_x
    global previous_y
    global agent_x
    global agent_y
    global move_x
    global move_y
    global next_x
    global next_y

    for i in range(1, max_steps):

        g.plotGraphBuilding(i)

        if i == 1:
            initializeMCV2D(previous_x, previous_y)

        status = plan2D()

        if status:
            print("all Node traversed")
            break

        previous_x = next_x
        previous_y = next_y

        agent_x = previous_x + move_x -2
        agent_y = previous_y + move_y -2

        manipulateGraph2D()

        foundcurrent = False
        foundprevious = False

        for v in g:
            vid = v.get_id()

            if vid == "("+str(previous_x)+","+str(previous_y)+")":
                foundprevious = True

            if vid == "("+str(agent_x)+","+str(agent_y)+")":
                foundcurrent = True

        if i > 1 and foundprevious and foundcurrent:
            vertex_agent = "(" + str(agent_x) + "," + str(agent_y) + ")"
            vertex_previous = "(" + str(previous_x) + "," + str(previous_y) + ")"
            g.add_edge(vertex_previous, g.get_vertex_mcv(vertex_previous), vertex_agent, g.get_vertex_mcv(vertex_agent), 1)







def initial_graph():

    global graph

    for v in g:
        temp = {}
        vid = v.get_id()
        for w in v.get_connections():
            wid = w.get_id()
            temp[wid] = v.get_weight(w)
        graph[vid] = temp



def planTrajectory():

    global traversal_status
    global agent_x
    global agent_y
    global previous_x
    global previous_y
    global graph

    path = {}
    adj_node = {}
    queue = []
    trajectory = []

    keys = graph.keys()

    startDistanceMin = 100
    goalDistanceMin = 100
    initial = ''
    x = ''

    # find nearest neighbour to the start and goal position
    for key in keys:
        temp = key.split(",")
        temp_x = temp[0].split("(")
        key_x = int(temp_x[1])
        temp_y = temp[1].split(")")
        key_y = int(temp_y[0])

        startDistance = math.sqrt((agent_x-key_x)*(agent_x-key_x) + (agent_y-key_y)*(agent_y-key_y))
        goalDistance = math.sqrt((goal_x-key_x)*(goal_x-key_x) + (goal_y-key_y)*(goal_y-key_y))

        if startDistance < startDistanceMin:
            # set it as initial
            initial = key
            startDistanceMin = startDistance
        if goalDistance < goalDistanceMin:
            # set it as terminal
            x = key
            goalDistanceMin = goalDistance

    print("initial point ", initial, startDistanceMin, " goal point ", x, goalDistanceMin)


    for node in graph:
        path[node] = float("inf")
        adj_node[node] = None
        queue.append(node)

    path[initial] = 0

    while queue:
        # find min distance which wasn't marked as current
        key_min = queue[0]
        min_val = path[key_min]
        for n in range(1, len(queue)):
            if path[queue[n]] < min_val:
                key_min = queue[n]
                min_val = path[key_min]
        cur = key_min
        queue.remove(cur)

        for i in graph[cur]:
            alternate = graph[cur][i] + path[cur]
            if path[i] > alternate:
                path[i] = alternate
                adj_node[i] = cur

    trajectory.append("("+str(goal_x)+","+str(goal_y)+")")
    trajectory.append(x)
    while True:
        x = adj_node[x]
        if x is None:
            trajectory.append("("+str(agent_x)+","+str(agent_y)+")")
            break
        trajectory.append(x)

    return trajectory





if __name__ == '__main__':

    max_episode = 10
    max_steps = 1000

    # ###################################################################################
    # ################################### IMPORTANT #####################################
    # ###################################################################################
    # enable below for boundary mask
    # needed one time, afterwards in update inner values are zero, so nothing matters
    boundarySwitch = True

    # enable below to update (Raise) MCV for previous position in the direction of movement
    fDirectionSwitch = True

    # enable below to update (Lower) MCV for current position from the direction of movement
    bDirectionSwitch = True

    agent_x, agent_y = randomValidPoint("FullGrid")

    # set history at initilisation
    previous_x, previous_y = agent_x, agent_y

    # return when goal is found or 1000 steps
    buildGraph(max_steps)

    initial_graph()

    print("GRAPH\n", graph)

    for t in trange(0, max_episode):

        agent_x, agent_y = randomValidPoint("FullGrid")
        goal_x, goal_y = randomValidPoint("FullGrid")

        # set history at initilisation
        previous_x, previous_y = agent_x, agent_y

        # return when goal is found or 1000 steps
        trajectory = planTrajectory()
        print("The path between ", "(" + str(agent_x) + "," + str(agent_y) + ")", " to ",
              "(" + str(goal_x) + "," + str(goal_y) + ")", trajectory)
        g.plotTrajectory(trajectory, t)
