# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
##my code
#from collections import deque

import util


#mapping = {'East':e,'West':w,'South':s,'North':n}
#mapping1 = {'East':'e','West':'w','South':'s','North':'n'}


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    REVERSE_PUSH = False

    @staticmethod
    def reverse_push():
        SearchProblem.REVERSE_PUSH = not SearchProblem.REVERSE_PUSH

    @staticmethod
    def print_push():
        print(SearchProblem.REVERSE_PUSH)

    @staticmethod
    def get_push():
        return SearchProblem.REVERSE_PUSH

    def get_expanded(self):
        return self.__expanded

    def inc_expanded(self):
        self.__expanded+=1

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

'''def rec_dfs1(state,problem,ls):
    ##
    
    
    if(problem.isGoalState(state)):
        if(state[1] == 'West'):
            return([True,ls])
        elif(state[1] == 'East'):
            return([True,ls])
        elif(state[1] == 'North'):
            return([True,ls])
        elif(state[1] == 'South'):
            return([True,ls])
        else:
            return([False,ls])
    


    else:
        successors = problem.getSuccessors(state)
        i = 0
        l = len(successors)
        while(i<l):
            next_state = successors[i]
            if(not(next_state in visited)):
                visited.append(next_state)
                x = rec_dfs1(next_state[0],problem,ls[1].append(mapping[next_state[1]]))
                ##x = rec_dfs(next_state[0],problem,[])
                if(x[0]==True):
                    return([True,x[1]])

            i = i + 1

        return([False,['here']])'''

def rec_dfs(state,problem,ls,visited):
    ##
    
    
    #print(state)
    #print("..")
    if(problem.isGoalState(state)):
        #visited.append(state)
        print("goal reached")
        return(ls)

    
    else:
        successors = problem.getSuccessors(state)
        i = 0
        l = len(successors)
        #print("num of successors : " + str(l))
        visited.append(state)
        #print("appending "+str(state)+" to visited")
        #print(visited)
        
        while(i<l):
            temp = ls
            next_state = successors[i]
            #print("state neighbour : "+str(next_state) + " " + str(next_state[0] in visited))

            if(not(next_state[0] in visited)):
                #print("visiting this state :")
                #print(next_state)
                t = next_state[1]
                
                temp.append(t)
                x = rec_dfs(next_state[0],problem,temp,visited)
                if(x==temp):
                    #print("found")
                    #print(next_state)
                    #print(temp)
                    #temp.pop()
                    return(x)
                else:
                    c = temp.pop()
                    #temp = ls
            i = i + 1
        return([])


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST
    "*** YOUR CODE HERE ***"
    
    #visited = []
    current_state = problem.getStartState()
    dfs_stack = [current_state]
    x = rec_dfs(current_state,problem,[],[])
    #while(visited != []):
    #   i = visited.pop()
    #print("sol is")
    #print(x)
    
    #print("Start:", problem.getStartState())

    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    return(x)
    #return([s,s,w,s,w,w,s,w])
    #['South', 'South', 'West', 'South', 'West', 'West', 'North', 'West', 'North', 'North', 'East', 'East', 'East', 'South', 'West']

    util.raiseNotDefined()


















def rec_bfs(state,problem):
    if(problem.isGoalState(state)):
        return(True)
    else:

        successors = problem.getSuccessors(state)
        i = 0
        l = len(successors)
        while(i<l):
            curr = successors[i][0]
            if(not(successors[i][0] in visited)):
                t = rec_bfs(curr,problem)
                if(t==True):
                    return(True)
            i=i+1
        return(False)


def breadthFirstSearch_m(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST
    print("inside bfs!!!!!!!!!!!!!!!!!!!!!!!")
    current_state = problem.getStartState()
    #print(current_state.get_vis())
    #print(current_state.get_state())
    bfs_queue = util.Queue()
    curr = current_state
    bfs_queue.push(curr)
    path = []
    visited = []
    parent = {}
    

    paths = {}
    paths[current_state] = []
    indicator = False
    check = current_state
    while(not(bfs_queue.isEmpty()) and not(indicator)):

        current_state = bfs_queue.pop()
        #print(current_state.get_state())
        #print(paths[current_state])
        current_state_t = current_state
        visited.append(current_state_t)
        
        
        if(problem.isGoalState(current_state)):
            indicator = True
            check = current_state
            return(paths[current_state])

        else:
            successors = problem.getSuccessors(current_state)
            l = len(successors)
            i = 0
            while(i<l):
                if(not(successors[i][0] in visited)):
                    visited.append(successors[i][0])
                    bfs_queue.push(successors[i][0])
                    t = paths[current_state] + [successors[i][1]]
                    #t.append(successors[i][1])
                    #ef = t
                    paths[successors[i][0]] = t
                    #c = t.pop()
                    
                i=i+1

    x = paths[check]
    #print(x)
    return(x)
    #return([s,s,w,s,w,w,s,w])
    util.raiseNotDefined()



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST
    #print("inside bfs!!!!!!!!!!!!!!!!!!!!!!!")
    current_state = problem.getStartState()
    #print(current_state.get_vis())
    #print(current_state.get_state())
    bfs_queue = util.Queue()
    curr = current_state
    bfs_queue.push((curr,[]))
    path = []
    visited = []
    parent = {}
    

    paths = {}
    #paths[current_state] = []
    indicator = False
    check = current_state
    while(not(bfs_queue.isEmpty()) and not(indicator)):

        (current_state,p) = bfs_queue.pop()
        #print(current_state.get_state())
        #print(paths[current_state])
        current_state_t = current_state
        visited.append(current_state_t)
        
        
        if(problem.isGoalState(current_state)):
            indicator = True
            check = current_state
            return(p)

        else:
            successors = problem.getSuccessors(current_state)
            l = len(successors)
            i = 0
            while(i<l):
                if(not(successors[i][0] in visited)):
                    visited.append(successors[i][0])
                    
                    t = p + [successors[i][1]]
                    bfs_queue.push((successors[i][0],t))
                    #t.append(successors[i][1])
                    #ef = t
                    #paths[successors[i][0]] = t
                    #c = t.pop()
                    
                i=i+1

    
    #print(x)
    #return(x)
    #return([s,s,w,s,w,w,s,w])
    util.raiseNotDefined()



























def make_list(q):
    
    l = []
    while(not(q.isEmpty())):
        x = q.pop()
        l.append(x)

    return(l)

def make_q(l1,l2):
    q = util.PriorityQueue()
    l = len(l1)
    i = 0
    while(i<l):
        q.push(l1[i],l2[l1[i]])
        i=i+1

    return(q)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST
    current_stat = problem.getStartState()
    
    visited = []
    parent = {}
    paths = {}
    paths[current_stat] = []
    costs = {}
    costs[current_stat] = 0
    indicator = False
    pd = []
    pd.append(current_stat)
    

    frontier = util.PriorityQueue()
    frontier.push(current_stat,0)
    
    explored = []

    while(not(frontier.isEmpty()) and not(indicator)):
        current_state = frontier.pop()
        #print("current state is :")
        #print(current_state)
        if(problem.isGoalState(current_state)):
            indicator = True
            print("goal found")
            return(paths[current_state])


        #print("here")
        explored.append(current_state)
        successors = problem.getSuccessors(current_state)
        #print("its neighbours are :")
        #print(successors)

        i = 0
        l = len(successors)
        while(i<l):
            #frontier_list = make_list(frontier)
            #print(frontier_list)
            temp = successors[i][0]
            temp_cost = costs[current_state] + successors[i][2]
            temp_path = paths[current_state] + [successors[i][1]]
            #print(temp_cost)
            #print(temp_path)
            if((not(temp in explored)) and (not(temp in pd))):
                costs[temp] = temp_cost
                paths[temp] = temp_path
                #print("doffffffffffffff" + str(temp_cost))
                '''frontier_list1 = frontier_list + [temp]
                frontier = make_q(frontier_list1,costs)'''
                frontier.push(temp,temp_cost)
                pd.append(temp)

            elif((temp in pd) and (costs[temp] > temp_cost)):
                '''indi = frontier_list.index(temp)
                del frontier_list[indi]'''
                costs[temp] = temp_cost
                paths[temp] = temp_path
                '''frontier_list1 = frontier_list + [temp]'''
                #frontier = make_q(frontier_list,costs)
                frontier.update(temp,temp_cost)

            #tempo = make_list(frontier)
            #print(tempo)
            i=i+1


    sol = paths[current_state]
    #print(sol)
    return(sol)
    #return(breadthFirstSearch(problem))
    util.raiseNotDefined()




def uniformCostSearch1(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST
    current_state = problem.getStartState()
    
    visited = []
    parent = {}
    paths = {}
    paths[current_state] = []
    costs = {}
    costs[current_state] = 0
    indicator = False
    check = current_state
    

    frontier = util.PriorityQueue()
    frontier.push(current_state,0)
    
    explored = []

    while(not(frontier.isEmpty()) and not(indicator)):
        current_state = frontier.pop()
        #print("current state is :")
        #print(current_state)
        if(problem.isGoalState(current_state)):
            indicator = True
            print("goal found")
            return(paths[current_state])


        #print("here")
        explored.append(current_state)
        successors = problem.getSuccessors(current_state)
        #print("its neighbours are :")
        #print(successors)

        i = 0
        l = len(successors)
        while(i<l):
            frontier_list = make_list(frontier)
            #print(frontier_list)
            temp = successors[i][0]
            temp_cost = costs[current_state] + successors[i][2]
            temp_path = paths[current_state] + [successors[i][1]]
            print(temp_cost)
            #print(temp_path)
            if((not(temp in explored)) and (not(temp in frontier_list))):
                costs[temp] = temp_cost
                paths[temp] = temp_path
                #print("doffffffffffffff" + str(temp_cost))

                frontier.push(temp,temp_cost)

            elif((temp in frontier_list) and (costs[temp] > temp_cost)):
                '''indi = frontier_list.index(temp)
                del frontier_list[indi]
                costs[temp] = temp_cost
                paths[temp] = temp_path
                frontier_list.append(temp)
                frontier = make_q(frontier_list,costs)'''
                frontier.update(temp,temp_cost)

            #tempo = make_list(frontier)
            #print(tempo)
            i=i+1


    sol = paths[current_state]
    #print(sol)


    


    return(sol)
    #return(breadthFirstSearch(problem))
    util.raiseNotDefined()





























def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def make_q1(l1,l2,h,p):
    q = util.PriorityQueue()
    l = len(l1)
    i = 0
    while(i<l):
        q.push(l1[i],(l2[l1[i]]+h(l1[i],p)))
        i=i+1
    return(q)


def aStarSearch1(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    current_state = problem.getStartState()
    #bfs_queue = PriorityQueue()
    #bfs_queue.put((0,current_state))
    #path = []
    pd = []
    visited = []
    parent = {}
    paths = {}
    paths[current_state] = []
    costs = {}
    costs[current_state] = 0 + heuristic(current_state,problem)
    indicator = False
    check = current_state

    frontier = util.PriorityQueue()
    frontier.push(current_state,(0+heuristic(current_state,problem)))
    pd.append(current_state)
    
    explored = []

    while(not(frontier.isEmpty()) and not(indicator)):
        current_state = frontier.pop()
        
        if(problem.isGoalState(current_state)):
            indicator = True
            print("goal found")
            return(paths[current_state])


        #print("here")
        explored.append(current_state)
        successors = problem.getSuccessors(current_state)
        

        i = 0
        l = len(successors)
        while(i<l):
            #frontier_list = make_list(frontier)
            #print(frontier_list)
            temp = successors[i][0]
            temp_cost = costs[current_state] + successors[i][2]
            temp_path = paths[current_state] + [successors[i][1]]
            #print(temp_cost)
            #print(temp_path)
            if((not(temp in explored)) and (not(temp in pd))):
                costs[temp] = temp_cost
                paths[temp] = temp_path
                
                frontier.push(temp,temp_cost+heuristic(temp,problem))
                pd.append(temp)
                #rontier = make_q1(frontier_list,costs,heuristic,problem)

            elif((temp in pd) and (costs[temp] > temp_cost)):
                
                costs[temp] = temp_cost
                paths[temp] = temp_path
                frontier.update(temp,temp_cost+heuristic(temp,problem))


            #tempo = make_list(frontier)
            #print(tempo)
            i=i+1


    sol = paths[current_state]
    #print(sol)
    return(sol)


    util.raiseNotDefined()

def get_path(z,q):
    i = 0
    while(i<len(q)):
        (x,y) = q[i]
        if(x==z):
            return(y)
        i=i+1


def pd1(pd):
    l = []
    i = 0
    while(i<len(pd)):
        (x,y) = pd[i]
        l.append(x)
        i=i+1
    return(l)


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    current_state = problem.getStartState()
    #bfs_queue = PriorityQueue()
    #bfs_queue.put((0,current_state))
    #path = []
    pd = []
    visited = []
    parent = {}
    paths = {}
    #paths[current_state] = []
    costs = {}
    #costs[current_state] = 0 + heuristic(current_state,problem)
    indicator = False
    check = current_state

    frontier = util.PriorityQueue()
    frontier.push(current_state,(0+heuristic(current_state,problem)))
    pd.append((current_state,[]))
    
    explored = []

    while(not(frontier.isEmpty()) and not(indicator)):
        current_state = frontier.pop()
        p = get_path(current_state,pd)
        
        if(problem.isGoalState(current_state)):
            indicator = True
            print("goal found")
            return(p)


        #print("here")
        explored.append(current_state)
        successors = problem.getSuccessors(current_state)
        

        i = 0
        l = len(successors)
        while(i<l):
            #frontier_list = make_list(frontier)
            #print(frontier_list)
            temp = successors[i][0]
            temp_cost = problem.getCostOfActions(p) + successors[i][2]
            temp_path = p + [successors[i][1]]
            #print(temp_cost)
            #print(temp_path)
            if((not(temp in explored)) and (not(temp in pd1(pd)))):
                #costs[temp] = temp_cost
                #paths[temp] = temp_path
                
                frontier.push(temp,temp_cost+heuristic(temp,problem))
                pd.append((temp,temp_path))
                #rontier = make_q1(frontier_list,costs,heuristic,problem)

            elif((temp in pd1(pd)) and (problem.getCostOfActions(get_path(temp,pd)) > temp_cost)):
                
                costs[temp] = temp_cost
                paths[temp] = temp_path
                frontier.update(temp,temp_cost+heuristic(temp,problem))
                prev_path = get_path(temp,pd)
                pd.remove((temp,prev_path))
                pd.append((temp,temp_path))


            #tempo = make_list(frontier)
            #print(tempo)
            i=i+1


    sol = paths[current_state]
    #print(sol)
    return(sol)


    util.raiseNotDefined()




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
reverse_push=SearchProblem.reverse_push
print_push=SearchProblem.print_push
