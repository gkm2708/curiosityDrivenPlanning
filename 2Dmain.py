import gym
import gym_lmaze
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

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def printGraph(self):
        #print("Printing Current Graph")

        """
        for v in g:
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                #print('( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w)))
                print('( %s , %s, %s)'  % ( vid, wid, v.get_weight(w)))

        """
        """
        for v in g:
            print('g.vert_dict[%s]=%s' %(v.get_id(), g.vert_dict[v.get_id()]))
        """

        for v in g:
            vid = v.get_id()
            print('( %s , %s)'  % ( vid, v.get_mcv()))


    def plotMap2D(self, realgrid, t, step, bDirection, fDirection):
        #print("PlotMap ", t," ", step)

        global imageMCV
        imageMCV = np.zeros((realgrid,realgrid))
        imageMCVFull = np.zeros((realgrid,realgrid))
        imageGrid = np.zeros((realgrid,realgrid))
        pointMCV = np.zeros((realgrid,realgrid))

        imageExpanded = np.zeros((realgrid*7*3,realgrid*7*3))

        # part 3 : Full MCV Map
        # part 1 : Episode MCV Map

        for v in g:

            vid = v.get_id()
            vid = vid.split(",")

            vidx = vid[0].split("(")
            vidy = vid[1].split(")")

            currentPointMCV = v.get_mcv()
            #print(vidx[1], vidy[0])

            begin_x = int(vidx[1])-2
            end_x = int(vidx[1])+3

            begin_y = int(vidy[0])-2
            end_y = int(vidy[0])+3

            #print(temp_goal_x, temp_goal_y, goal_x, goal_y)
            #print("success")

            for i in range(begin_x, end_x):
                #print("i ",i)
                for j in range(begin_y, end_y):
                    #print("j ",j)
                    #print("i ", i,
                    #      "j ", j,
                    #      "int(vidx[1])", int(vidx[1]),
                    #      "int(vidy[0])", int(vidy[0]),
                    #      "reference to zero ", i-begin_x,
                    #      "reference to zero ", j-begin_y)
                    imageMCVFull[i, j] = imageMCVFull[i, j] + currentPointMCV[i-begin_x, j-begin_y]

        #print(imageMCVFull)
        x, y = np.unravel_index(imageMCVFull.argmax(), imageMCVFull.shape)
        imageMCVFull = imageMCVFull*255/imageMCVFull[x][y]

        # part 4 : Full Grid Map

        imageGrid = [ [0 if item == 0 else 255 for item in row] for row in grid]
        imageGrid[previous_x][previous_y] = 200
        imageGrid[agent_x][agent_y] = 150
        imageGrid[goal_x][goal_y] = 100


        # part 2 : Instantaneous MCV Map
        vertex = "("+str(agent_x)+","+str(agent_y)+")"
        tempMCV = g.get_vertex_mcv(vertex)

        #print(t, step)
        #print(tempMCV)

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

        cv2.imwrite("2d/"+str(t)+"/curiosity_map_"+str(t)+"_"+str(step)+".png", imageExpanded)















g = Graph()

# initialize grid
grid = np.array([   [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
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


realgrid = grid.shape[0]

boundarymask = np.array([   [1, 1, 1, 1, 1],
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
previous_to_previous_x, previous_to_previous_y = 0, 0

boundarySwitch = False
fDirectionSwitch = False
bDirectionSwitch = False





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

    vertex = "("+str(agent_x)+","+str(agent_y)+")"
    tempMCV = g.get_vertex_mcv(vertex)

    #print(tempMCV)
    #print("Plan2D ", vertex)

    unique = np.unique(tempMCV, return_counts=True)
    #print("\nUnique ",unique)

    best = unique[0][-1]
    #print("Best ", best)

    count = unique[1][-1]
    #print("Count", count)

    # find if highest curiosity has multiple counts
    # if yes : action random
    # acts as a tiebreaker for initialized MCV and randomize exact move in the direction
    if count > 1:
        searchIndex = 0
        actionIndex = random.randint(1,count)
        #print("chosen action ", actionIndex, " between  ", 1,", ", count)
        for i in range(0,5):
            for j in range(0,5):
                if tempMCV[i][j] == best:
                    searchIndex +=1
                if searchIndex == actionIndex:
                    #print(actionIndex, searchIndex)
                    move_x = i
                    move_y = j
                    break
            if searchIndex == actionIndex:
                break
    else: # else  : max curiosity
        #print("Best value is ", best)
        move_x, move_y = np.unravel_index(tempMCV.argmax(), tempMCV.shape)







"""    
"""
def initializeMCV2D(t,step):

    global previous_x
    global previous_y
    global agent_x
    global agent_y
    global move_x
    global move_y
    global goal_x
    global goal_y

    #print("initialize ", t, step)

    # add this point as vertex
    vertex = "("+str(agent_x)+","+str(agent_y)+")"

    #print("InitializeMCD2D ", vertex)

    # hardcoded reachability discovery from grid
    #
    reachability = grid[agent_x-2:agent_x+3, agent_y-2:agent_y+3]

    #print(agent_x, agent_y, reachability)

    # Initialize MCV
    tempMCV = initMCV * reachability

    # mask boundary
    if boundarySwitch:
        tempMCV = tempMCV * boundarymask

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

    vertex_agent = "("+str(agent_x)+","+str(agent_y)+")"
    vertex_previous = "("+str(previous_x)+","+str(previous_y)+")"

    #print("\nUpdateMCV2D", vertex_agent, "\n")

    tempMCV_agent = g.get_vertex_mcv(vertex_agent)
    tempMCV_previous = g.get_vertex_mcv(vertex_previous)

    bMask = np.zeros((realgrid,realgrid))
    fMask = np.zeros((realgrid,realgrid))

    bMask[agent_x-2:agent_x+3,agent_y-2:agent_y+3] = 1.0
    fMask[previous_x-2:previous_x+3,previous_y-2:previous_y+3] = 1.0

    if previous_x <= agent_x:
        #print("Condition 1 or 2")
        if previous_y <= agent_y:
            #print("Condition 1 ")
            bMask[previous_x:agent_x+1, previous_y:agent_y+1] = 0.99
            fMask[previous_x:agent_x+1, previous_y:agent_y+1] = 1.01
        else:
            #print("Condition 2")
            bMask[previous_x:agent_x+1, agent_y:previous_y+1] = 0.99
            fMask[previous_x:agent_x+1, agent_y:previous_y+1] = 1.01
    else:
        #print("Condition 3 or 4")
        if previous_y <= agent_y:
            #print("Condition 3 ")
            bMask[agent_x:previous_x+1, previous_y:agent_y+1] = 0.99
            fMask[agent_x:previous_x+1, previous_y:agent_y+1] = 1.01
        else:
            #print("Condition 4 ")
            bMask[agent_x:previous_x+1, agent_y:previous_y+1] = 0.99
            fMask[agent_x:previous_x+1, agent_y:previous_y+1] = 1.01


    bDirection = bMask[agent_x-2:agent_x+3,agent_y-2:agent_y+3]
    fDirection = fMask[previous_x-2:previous_x+3,previous_y-2:previous_y+3]

    tempMCV_agent = tempMCV_agent * bDirection
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
def manipulateGraph2D(t,step):

    #print("ManipulateGraph2D")

    global previous_x
    global previous_y
    global agent_x
    global agent_y
    global move_x
    global move_y
    global goal_x
    global goal_y

    found = False

    bDirection = np.zeros((realgrid,realgrid))
    fDirection = np.zeros((realgrid,realgrid))

    if g.num_vertices > 0:
        # Non Empty Graph
        found = False
        for v in g:
            vid = v.get_id()
            if vid == "("+str(agent_x)+","+str(agent_y)+")":
                #print("Manipulate Update episode ", t, ", step " , step)

                # node found hence update MCV and find next action
                bDirection, fDirection = updateMCV2D()
                found = True
                break

    if not found or g.num_vertices == 0:
        # node not found hence initialize MCV and find next action
        # or started with empty graph

        #print("manipulate Initialize episode ", t, ", step " , step)
        initializeMCV2D(t,step)

    #plan2D()
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

    #print("rollout2D")

    for i in range(1, max_steps):
        #print("Episode ", t, " Step ", i)
        bDirection, fDirection = manipulateGraph2D(t,i)

        previous_x = agent_x
        previous_y = agent_y

        plan2D()
        g.plotMap2D(realgrid,t,i, bDirection, fDirection)

        # Not Moving : Out of Boundary
        #print("checking boundary conditions")
        #print(1, agent_x + move_x -2, realgrid - 2, 1, agent_y + move_y -2, realgrid - 2)
        if 1 < agent_x + move_x -2 < realgrid - 2 and 1 < agent_y + move_y -2 < realgrid - 2:
            #print("successful move")
            agent_x = agent_x + move_x - 2
            agent_y = agent_y + move_y - 2


        # If Goal Found
        #print("checking Goal conditions")
        #print( agent_x - 3, goal_x, agent_x + 3, agent_y - 3, goal_y, agent_y + 3)
        if 1 < agent_x < realgrid - 2 and 1 < agent_y < realgrid - 2:
            if agent_x - 3 < goal_x < agent_x + 3 and agent_y - 3 < goal_y < agent_y + 3:
                #print("Goal Found")
                break

        #time.sleep(1)










if __name__ == '__main__':

    max_episode = 100
    max_steps = 100


    ##################### IMPORTANT ######################
    # enable below for boundary mask
    # needed one time, afterwards in update inner values are zero, so nothing matters
    boundarySwitch = True


    # enable below to update (Raise) MCV for previous position in the direction of movement
    fDirectionSwitch = False


    # enable below to update (Lower) MCV for current position from the direction of movement
    bDirectionSwitch = True

    # Trial
    for t in trange(0, max_episode):
        if not os.path.exists("2d/"+str(t)):
            os.makedirs("2d/"+str(t))

        # Initialize agent and Goal
        #
        # (SmallGrid (fixed samples of start and goal points; required to test 4D(start and goal, x and y)) /
        #
        # FullGrid (start and goal point sampling from any valid point))
        #
        agent_x, agent_y = randomValidPoint("SmallGrid")
        goal_x, goal_y = randomValidPoint("SmallGrid")


        # set history at initilisation
        previous_x, previous_y = agent_x, agent_y

        # return when goal is found or 1000 steps
        rollout2D(t, max_steps)

    g.printGraph()
