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
        imageGridFull = np.zeros((realgrid*2,realgrid*2))

        imageExpanded = np.zeros((realgrid*7,realgrid*2*7))

        for v in g:

            vid = v.get_id()
            vid = vid.split(",")

            vidx = vid[0].split("(")
            vidy = vid[1].split(")")

            x = int(vidx[1])
            y = int(vidy[0])

            imageMCV[x][y] = 255

        #temp = [[0 if item == 0 else 255 for item in row] for row in grid]
        imageGridFull = np.concatenate([imageMCV, [[0 if item == 0 else 255 for item in row] for row in grid] ], axis=1)

        for i in range(0, imageGridFull.shape[0]):
            for ii in range(0, 7):
                for j in range(0, imageGridFull.shape[1]):
                    for jj in range(0, 7):
                        imageExpanded[i * 7 + ii][j * 7 + jj] = imageGridFull[i][j]

        cv2.imwrite("curiosity_map.png", imageExpanded)


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
traversal_status = {}

boundarySwitch = False
fDirectionSwitch = False
bDirectionSwitch = False

"""    
"""

def initializeMCV2D(point_x, point_y):

    global traversal_status
    global boundarySwitch

    # add this point as vertex
    vertex = "(" + str(point_x) + "," + str(point_y) + ")"
    # hardcoded reachability discovery from grid
    reachability = grid[point_x-2:point_x+3, point_y-2:point_y+3]
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

    tempMCV_agent = g.get_vertex_mcv(vertex_agent)
    tempMCV_previous = g.get_vertex_mcv(vertex_previous)

    bMask = np.zeros((realgrid,realgrid))
    fMask = np.zeros((realgrid,realgrid))

    bMask[agent_x-2:agent_x+3,agent_y-2:agent_y+3] = 1.0
    fMask[previous_x-2:previous_x+3,previous_y-2:previous_y+3] = 1.0

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

    bDirection = bMask[agent_x-2:agent_x+3,agent_y-2:agent_y+3]
    fDirection = fMask[previous_x-2:previous_x+3,previous_y-2:previous_y+3]

    tempMCV_agent = tempMCV_agent * bDirection
    tempMCV_previous = tempMCV_previous * fDirection

    # enable below to lower curiosity from backward direction
    if bDirectionSwitch:
        g.set_vertex_mcv(vertex_agent, tempMCV_agent)
        unique = np.unique(tempMCV_agent, return_counts=True)
        if unique[0].shape[0] == 1:
            traversal_status[vertex_agent] = "Done"

    # enable below to raise curiosity in forward direction
    if fDirectionSwitch:
        g.set_vertex_mcv(vertex_previous, tempMCV_previous)
        unique = np.unique(tempMCV_previous, return_counts=True)
        if unique[0].shape[0] == 1:
            traversal_status[vertex_previous] = "Done"

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
        initializeMCV2D(agent_x, agent_y)
    if not foundprevious:
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
                print("found non zero action at ", v, " action is ", next_x+move_x-2, next_y+move_y-2)
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

        if i > 1:
            vertex_agent = "(" + str(agent_x) + "," + str(agent_y) + ")"
            vertex_previous = "(" + str(previous_x) + "," + str(previous_y) + ")"
            g.add_edge(vertex_previous, g.get_vertex_mcv(vertex_previous), vertex_agent, g.get_vertex_mcv(vertex_agent), 1)

    g.printGraph()










def planTrajectory():

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
                print("found non zero action at ", v, " action is ", next_x+move_x-2, next_y+move_y-2)
                break
        if findStatus:
            break
    if not findStatus:
        print("finished building graph")
        return True
    return False





if __name__ == '__main__':

    max_episode = 1
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
    bDirectionSwitch = False



    #agent_x, agent_y = randomValidPoint("FullGrid")
    agent_x, agent_y = 2,2
    goal_x, goal_y = randomValidPoint("SmallGrid")

    # set history at initilisation
    previous_x, previous_y = agent_x, agent_y

    # return when goal is found or 1000 steps
    buildGraph(max_steps)
    g.plotGraph()
    g.printGraph()





    for t in trange(0, max_episode):
        if dimensionCount == 2:
            if not os.path.exists("2d/"+str(t)):
                os.makedirs("2d/"+str(t))
        elif dimensionCount == 4:
            if not os.path.exists("4d/"+str(t)):
                os.makedirs("4d/"+str(t))

        agent_x, agent_y = randomValidPoint("FullGrid")
        goal_x, goal_y = randomValidPoint("FullGrid")

        # set history at initilisation
        previous_x, previous_y = agent_x, agent_y

        # return when goal is found or 1000 steps
        trajectory = planTrajectory()
