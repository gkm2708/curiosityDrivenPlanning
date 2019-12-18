import numpy as np
from tqdm import trange
import random
import os
import time
import cv2
np.set_printoptions(threshold=np.inf)

""" ###################### VERTEX ####################"""

class Vertex:
    def __init__(self, node, mat):
        #print("Creating Vertex \n", mat)
        self.id = node
        self.mcv = mat
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def get_adjacent(self):
        return [x.id for x in self.adjacent]

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_mcv(self):
        #print("While getting \n",self.mcv)
        return self.mcv

    def set_mcv(self, tempMCV):
        #print("While setting \n",tempMCV)
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
        #print("Inside add vertex \n", mat)
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
            #print("get vertex mcv \n", self.vert_dict[n].get_mcv())
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

        """
        for v in g:
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                #print('( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w)))
                print('( %s , %s, %s)'  % ( vid, wid, v.get_weight(w)))

        """

        #"""
        for v in g:
            print('g.vert_dict[%s]=%s' %(v.get_id(), g.vert_dict[v.get_id()]))
        #"""

        """
        for v in g:
            vid = v.get_id()
            print('( %s , \n%s)' % (vid, v.get_mcv()))
        """

    def plotMap2D(self, t, step, bDirection, fDirection):

        global imageMCV
        imageMCV = np.zeros((realgrid,realgrid))
        imageMCVFull = np.zeros((realgrid,realgrid))
        #imageGrid = np.zeros((realgrid,realgrid))
        pointMCV = np.zeros((realgrid,realgrid))

        imageExpanded = np.zeros((realgrid*7*3, realgrid*7*3))

        # part 3 : Full MCV Map
        # part 1 : Episode MCV Map

        for v in g:

            vid = v.get_id()
            vid = vid.split(",")

            vidx = vid[0].split("(")

            if dimensionCount == 2:
                vidy = vid[1].split(")")
            elif dimensionCount == 4:
                vidy = vid[1]

            currentPointMCV = v.get_mcv()

            begin_x = int(vidx[1])-2
            end_x = int(vidx[1])+3

            if dimensionCount == 2:
                begin_y = int(vidy[0])-2
                end_y = int(vidy[0])+3
            elif dimensionCount == 4:
                begin_y = int(vidy)-2
                end_y = int(vidy)+3
                temp_goal_x = vid[2]
                temp_temp_goal_y = vid[3].split(")")
                temp_goal_y = temp_temp_goal_y[0]

            for i in range(begin_x, end_x):
                for j in range(begin_y, end_y):
                    if dimensionCount == 4 and int(temp_goal_x) == int(goal_x) and int(temp_goal_y) == int(goal_y):
                        imageMCV[i, j] = imageMCV[i, j] + currentPointMCV[i-begin_x, j-begin_y]
                    imageMCVFull[i, j] = imageMCVFull[i, j] + currentPointMCV[i-begin_x, j-begin_y]

        if dimensionCount == 4:
            x, y = np.unravel_index(imageMCV.argmax(), imageMCV.shape)
            imageMCV = imageMCV*255/imageMCV[x][y]

        x, y = np.unravel_index(imageMCVFull.argmax(), imageMCVFull.shape)
        imageMCVFull = imageMCVFull*255/imageMCVFull[x][y]

        # part 4 : Full Grid Map

        imageGrid = [[0 if item == 0 else 255 for item in row] for row in grid]
        imageGrid[previous_x][previous_y] = 200
        imageGrid[agent_x][agent_y] = 150
        imageGrid[goal_x][goal_y] = 100

        # part 2 : Instantaneous MCV Map
        if dimensionCount == 2:
            vertex = "("+str(agent_x)+","+str(agent_y)+")"
        elif dimensionCount == 4:
            vertex = "("+str(agent_x)+","+str(agent_y)+","+str(goal_x)+","+str(goal_y)+")"

        tempMCV = g.get_vertex_mcv(vertex)

        unique = np.unique(tempMCV, return_counts=True)

        multiplier = 255 / unique[0].shape[0]

        for i in range(0, 5):
            for j in range(0, 5):
                for k in range(0, unique[0].shape[0]):
                    if tempMCV[i][j] == unique[0][k]:
                        tempMCV[i][j] = k*multiplier

        pointMCV[agent_x-2:agent_x+3,agent_y-2:agent_y+3] = tempMCV

        bDirection = [[0 if item == 0 else 255 if item == 1 else 200 for item in row] for row in bDirection]
        fDirection = [[0 if item == 0 else 200 if item == 1 else 255 for item in row] for row in fDirection]

        imageFull = np.concatenate(
                    [np.concatenate([pointMCV, fDirection, bDirection], axis=1),
                    np.concatenate([imageMCV, imageMCVFull, imageGrid], axis=1)],
                    axis=0)

        for i in range(0, imageFull.shape[0]):
            for ii in range(0, 7):
                for j in range(0, imageFull.shape[1]):
                    for jj in range(0, 7):
                        imageExpanded[i * 7 + ii][j * 7 + jj] = imageFull[i][j]

        if dimensionCount == 2:
            cv2.imwrite("2d/"+str(t)+"/curiosity_map_"+str(t)+"_"+str(step)+".png", imageExpanded)
        if dimensionCount == 4:
            cv2.imwrite("4d/"+str(t)+"/curiosity_map_"+str(t)+"_"+str(step)+".png", imageExpanded)















g = Graph()

# initialize grid
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
bDirection = np.zeros((realgrid, realgrid))
fDirection = np.zeros((realgrid, realgrid))

boundarySwitch = False
fDirectionSwitch = False
bDirectionSwitch = False
dimensionCount = 2













"""    
"""


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















"""    
"""


def plan2D():

    global previous_x
    global previous_y
    global agent_x
    global agent_y
    global move_x
    global move_y
    global goal_x
    global goal_y
    global bDirection
    global fDirection

    if dimensionCount == 2:
        vertex = "(" + str(agent_x) + "," + str(agent_y) + ")"
    elif dimensionCount == 4:
        vertex = "(" + str(agent_x) + "," + str(agent_y) + "," + str(goal_x) + "," + str(goal_y) + ")"

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
                        move_x = i
                        move_y = j
                        break
                if searchIndex == actionIndex:
                    break
        else:  # else  : max curiosity
            move_x, move_y = np.unravel_index(tempMCV.argmax(), tempMCV.shape)
    else:
        print("\nonly zero curiosity left at this node : find a connected node with non zero value")
        for v in g:
            if str(v.get_id()) == vertex:
                temp = v.get_adjacent()
                for i in range(0, len(temp)):
                    print(temp[i])
        exit(0)















"""    
"""


def initializeMCV2D(t, step):

    global previous_x
    global previous_y
    global agent_x
    global agent_y
    global move_x
    global move_y
    global goal_x
    global goal_y
    global bDirection
    global fDirection

    # add this point as vertex
    if dimensionCount == 2:
        vertex = "(" + str(agent_x) + "," + str(agent_y) + ")"
    elif dimensionCount == 4:
        vertex = "(" + str(agent_x) + "," + str(agent_y) + "," + str(goal_x) + "," + str(goal_y) + ")"

    # hardcoded reachability discovery from grid
    #
    reachability = grid[agent_x-2:agent_x+3, agent_y-2:agent_y+3]

    # Initialize MCV
    tempMCV = initMCV * reachability

    # mask boundary
    if boundarySwitch:
        tempMCV = tempMCV * boundarymask

    #print("Initialize \n",tempMCV)
    g.add_vertex(vertex, tempMCV)


"""    
"""


def updateMCV2D():

    global previous_x
    global previous_y
    global agent_x
    global agent_y
    global move_x
    global move_y
    global goal_x
    global goal_y
    global bDirection
    global fDirection

    if dimensionCount == 2:
        vertex_agent = "(" + str(agent_x) + "," + str(agent_y) + ")"
        vertex_previous = "(" + str(previous_x) + "," + str(previous_y) + ")"
    elif dimensionCount == 4:
        vertex_agent = "(" + str(agent_x) + "," + str(agent_y) + "," + str(goal_x) + "," + str(goal_y) + ")"
        vertex_previous = "(" + str(previous_x) + "," + str(previous_y) + "," + str(goal_x) + "," + str(goal_y) + ")"

    tempMCV_agent = g.get_vertex_mcv(vertex_agent)
    #print("Inside Update\n",tempMCV_agent)

    tempMCV_previous = g.get_vertex_mcv(vertex_previous)

    bMask = np.zeros((realgrid,realgrid))
    fMask = np.zeros((realgrid,realgrid))

    bMask[agent_x-2:agent_x+3,agent_y-2:agent_y+3] = 1.0
    fMask[previous_x-2:previous_x+3,previous_y-2:previous_y+3] = 1.0

    if previous_x <= agent_x:
        if previous_y <= agent_y:
            bMask[previous_x:agent_x+1, previous_y:agent_y+1] = 0.0
            fMask[previous_x:agent_x+1, previous_y:agent_y+1] = 1.01
        else:
            bMask[previous_x:agent_x+1, agent_y:previous_y+1] = 0.0
            fMask[previous_x:agent_x+1, agent_y:previous_y+1] = 1.01
    else:
        if previous_y <= agent_y:
            bMask[agent_x:previous_x+1, previous_y:agent_y+1] = 0.0
            fMask[agent_x:previous_x+1, previous_y:agent_y+1] = 1.01
        else:
            bMask[agent_x:previous_x+1, agent_y:previous_y+1] = 0.0
            fMask[agent_x:previous_x+1, agent_y:previous_y+1] = 1.01

    bDirection = bMask[agent_x-2:agent_x+3,agent_y-2:agent_y+3]
    fDirection = fMask[previous_x-2:previous_x+3,previous_y-2:previous_y+3]

    tempMCV_agent = tempMCV_agent * bDirection
    #print(tempMCV_agent)

    tempMCV_previous = tempMCV_previous * fDirection

    # enable below to lower curiosity from backward direction
    if bDirectionSwitch:
        g.set_vertex_mcv(vertex_agent, tempMCV_agent)

    # enable below to raise curiosity in forward direction
    if fDirectionSwitch:
        g.set_vertex_mcv(vertex_previous, tempMCV_previous)

    return bMask, fMask


"""    
"""


def manipulateGraph2D(t, step):

    global previous_x
    global previous_y
    global agent_x
    global agent_y
    global move_x
    global move_y
    global goal_x
    global goal_y
    global bDirection
    global fDirection

    found = False

    bDirection = np.zeros((realgrid,realgrid))
    fDirection = np.zeros((realgrid,realgrid))

    if g.num_vertices > 0:
        # Non Empty Graph
        found = False
        for v in g:
            vid = v.get_id()
            if dimensionCount == 2 and vid == "("+str(agent_x)+","+str(agent_y)+")":
                # node found hence update MCV and find next action
                bDirection, fDirection = updateMCV2D()
                found = True
                break
            elif dimensionCount == 4 and vid == "("+str(agent_x)+","+str(agent_y)+","+str(goal_x)+","+str(goal_y)+")":
                # node found hence update MCV and find next action
                bDirection, fDirection = updateMCV2D()
                found = True
                break

    if not found or g.num_vertices == 0:
        # node not found hence initialize MCV and find next action
        # or started with empty graph
        initializeMCV2D(t, step)

    return bDirection, fDirection
















"""    
"""


def rollout2D(t, max_steps):

    global previous_x
    global previous_y
    global agent_x
    global agent_y
    global move_x
    global move_y
    global goal_x
    global goal_y
    global bDirection
    global fDirection


    # first step : initialize first node
    #       else : check if intialization is needed or upate is needed
    #               also add an edge
    # plan for next node if curiosity guides
    # else traverse to reach a node in graph with non zero curiosity


    for i in range(1, max_steps):
        bDirection, fDirection = manipulateGraph2D(t, i)

        # assume that a next node is found
        # add a link from previous to this with zero value on edge

        if dimensionCount == 2:
            vertex_agent = "(" + str(agent_x) + "," + str(agent_y) + ")"
            vertex_previous = "(" + str(previous_x) + "," + str(previous_y) + ")"
        elif dimensionCount == 4:
            vertex_agent = "(" + str(agent_x) + "," + str(agent_y) + "," + str(goal_x) + "," + str(goal_y) + ")"
            vertex_previous = "(" + str(previous_x) + "," + str(previous_y) + "," + str(goal_x) + "," + str(
                goal_y) + ")"

        if i > 1:
            g.add_edge(vertex_previous, g.get_vertex_mcv(vertex_previous), vertex_agent, g.get_vertex_mcv(vertex_agent), 1)




        previous_x = agent_x
        previous_y = agent_y



        # plan to find either non zero curiosity or a connected node with non zero curiosity
        plan2D()

        #g.plotMap2D(t, i, bDirection, fDirection)

        agent_x = agent_x + move_x - 2
        agent_y = agent_y + move_y - 2



        # If Goal Found
        #if 1 < agent_x < realgrid - 2 and 1 < agent_y < realgrid - 2:
        #    if agent_x - 3 < goal_x < agent_x + 3 and agent_y - 3 < goal_y < agent_y + 3:
        #        break

        #time.sleep(10)


if __name__ == '__main__':

    max_episode = 1
    max_steps = 100

    # ###################################################################################
    # ################################### IMPORTANT #####################################
    # ###################################################################################
    # enable below for boundary mask
    # needed one time, afterwards in update inner values are zero, so nothing matters
    boundarySwitch = True

    # enable below to update (Raise) MCV for previous position in the direction of movement
    fDirectionSwitch = False

    # enable below to update (Lower) MCV for current position from the direction of movement
    bDirectionSwitch = True

    dimensionCount = 2

    # Trial
    for t in trange(0, max_episode):
        if dimensionCount == 2:
            if not os.path.exists("2d/"+str(t)):
                os.makedirs("2d/"+str(t))
        elif dimensionCount == 4:
            if not os.path.exists("4d/"+str(t)):
                os.makedirs("4d/"+str(t))

        # Initialize agent and Goal
        #
        # (SmallGrid (fixed samples of start and goal points; required to test 4D(start and goal, x and y)) /
        #
        # FullGrid (start and goal point sampling from any valid point))
        #
        agent_x, agent_y = randomValidPoint("FullGrid")
        goal_x, goal_y = randomValidPoint("SmallGrid")

        # set history at initilisation
        previous_x, previous_y = agent_x, agent_y

        # return when goal is found or 1000 steps
        rollout2D(t, max_steps)

        g.printGraph()
