#============ LIBRARIES ================
import Wumpus_World as ww
from IPython.display import clear_output
import time


# This is my A* pathing without using the god view (so I’m not looking at the whole
# board or Wumpus location directly). The idea is that the agent has to figure
# things out from what it can actually sense each turn (breeze/stench) and what it
# has already seen before. I keep track of “safe” spots and “visited” spots so
# I’m not just wandering in circles.
#
# How it works:
# - Start at the agent’s position. Look at each direction. If there’s no breeze
#   and no stench in that direction, I mark that tile as safe.
# - SAFE and VISITED are saved between moves. Frontier = SAFE minus VISITED.
# - Pick the closest frontier tile as the next subgoal (using Manhattan distance).
# - Use A* to get there, but only avoid actual hazards (pits, Wumpus tile).
#   If I think a tile might be risky from breeze/stench, I don’t block it —
#   I just give it a +3 cost so the agent will only go there if it has to.
# - Keep doing this until we get to the gold.

#============ A * HELPER FUNCTIONS ===============
#Function calculates the Manhattan distance between the agent and a frontier cell (r,c)
def Manhattan_distance(agent_position,frontier_position):
    return abs(agent_position[0] - frontier_position[0]) + abs(agent_position[1] - frontier_position[1])

#================ A* FUNCTION ====================
#A* function that plans a path it does two things
# -Block any hazard tile(pits and Wumpus)
# -if tile is breeze/stench it add a small penalty of +3
def a_star(board, start, subgoal, suspect=None, risk_cost = 3):
    if start == subgoal:
        return [start]

    steps_to_explore = []  #list of frontier tiles to explore, stores (f, g, cell)
    g_score = {}           #maps cell (r,c) -> g-cost (steps from start)
    g_score[start] = 0     #starting cell g-cost is 0
    closed = set()         #set of cells already visited prevents looping
    came_from = {}         #maps child cell -> parent cell to make path

    #Add the starting node to the frontier where g= 0 at start and h = distance of subgoal
    h = Manhattan_distance(start, subgoal)
    steps_to_explore.append((h, 0, start))

    #The exploration loops, it keeps going as long as there are frontier tiles
    while steps_to_explore:

        #Picks the tile with lowest f value
        current = min(steps_to_explore, key = lambda x: x[0])
        steps_to_explore.remove(current)    #remove it from the frontier list
        f, g, cell = current

        #Skips cell if its been explored
        if cell in closed:
            continue

        closed.add(cell)      #marks it as explored, so agent doesn't revisit it

        #if the goal has been reach, rebuild the path
        if cell == subgoal:
            path = []
            current = cell
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            return path


        r, c = cell

        #Checks all neighboring cells
        neighbors = board.get_neighbors(r, c)

        #This code is if you dont want agent to be in the stench it does use the god view
        #Which we are trying to eliminate, but still here if you want to use it. It
        #Causes bugs if the Wumpus starts next to the agent.pos cell.
        #stench_cells = set()
        #if board.wumpus_pos is not None:
            #stench_cells = set(board.get_neighbors(*board.wumpus_pos))

        #loop to check if all neighboring cells are a hazard
        for nr, nc in neighbors:
            #Blocks all tiles that are lethal
            if board.is_hazard(nr, nc):
                continue

            #This skips any cell that has stench in it.
            #if (nr, nc) in stench_cells:
                #continue

            #if cell isn't a hazard add 1 to g
            possible_step = g + 1

            #check if cell is already in our g_score dictionary and or if cell value has changed - in case wumpus moves
            if (nr, nc) not in g_score or possible_step < g_score[(nr, nc)]:
                g_score[(nr, nc)] = possible_step                                 #add the key to the position
                came_from[(nr, nc)] = cell                                        #add your current cell to came_from
                h = Manhattan_distance((nr, nc), subgoal)                         #Heuristic to subgoal

                #Risk penalty
                if (suspect is not None) and (nr, nc) in suspect:
                  risk = risk_cost
                else:
                  risk = 0

                f = possible_step + h + risk                                       #calculate the f score
                steps_to_explore.append((f, possible_step, (nr, nc)))             #adds the cell to frontier with its f value

    return None   #In case no path found

#============= MOVE FUNCTION HELPERS =================
#Given two adjacent cells, return the AgentDirection needed to face from start -> end.
def change_facing_direction(start, end):
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    dir_map = {
        (0, 1): ww.AgentDirection.right,
        (0, -1): ww.AgentDirection.left,
        (1, 0): ww.AgentDirection.down,
        (-1, 0): ww.AgentDirection.up,
    }
    try:
        return dir_map[(dr, dc)]
    except KeyError:
        raise ValueError("start and end must be adjacent orthogonal cells")

#Returns the AgentActions needed to face desired direction
def turns_needed(current_facing, desired_facing):
    diff = (desired_facing.value - current_facing.value) % 4
    if diff == 0:  # already facing
        return []
    if diff == 1:
        return [ww.AgentAction.turn_right]
    if diff == 3:
        return [ww.AgentAction.turn_left]
    return [ww.AgentAction.turn_right, ww.AgentAction.turn_right]

#Functions turns agent towards the Wumpus and shoots 
def try_shoot_wumpus(board, perception, animate=True, sleep=1):
    #check for arrows
    if board.num_arrows <= 0:
        return False

    #Get stench location in relations to agent
    rel_stench = getattr(perception, "stench", [False, False, False, False])

    #Check the neighboring cells in stench direction
    directions = [(0, +1), (+1, 0), (0, -1), (-1, 0)]  #right, down, left, up
    facing_val = board.agent_facing.value
    r0, c0 = board.agent_pos

    for i, stinks in enumerate(rel_stench):
        if not stinks:
            continue

        #Compute the neighbor cell with stench
        dr, dc = directions[(facing_val + i) % 4]
        nr, nc = r0 + dr, c0 + dc
        if not board.is_in_bounds(nr, nc):
            continue  #if out of bounds, skip

        #Turn toward that neighbor
        desired_facing = change_facing_direction((r0, c0), (nr, nc))
        for action in turns_needed(board.agent_facing, desired_facing):
            board.actuate(action)
            if animate:
                clear_output(wait=True)
                ww.print_board_text(board)
                time.sleep(sleep)

        #Fire the Arrow
        board.actuate(ww.AgentAction.shoot)

        #Print result if Wumpus was killed
        new_perceive = board.perceive()
        if getattr(new_perceive, "scream", False):
            print(">> Wumpus down! Path is clear to the gold.")
        else:
            print(">> Shot fired, but no scream...")

        if animate:
            clear_output(wait=True)
            ww.print_board_text(board)
            time.sleep(sleep)
        return True  #one shot per call

    return False

#================ AGENT MEMORY HELPER ================
#Initializes the agent's memory sets for visted and safe cells
def init_memory(start):
    visited = set()         #tiles the agent has already been on
    safe = set()            #tles the agent has determined are safe to enter
    visited.add(start)
    safe.add(start)
    return visited, safe

#================== MOVE FUNCTION ===================
# Moves the agent around the board and shows what’s going on.
# It does a few things here:
#  - stops if we hit the max number of steps
#  - shows the board if animate=True
#  - sleep controls how fast the board updates (1 sec by default)
def move_agent(board, goal, visited, safe, max_steps=200, sleep=2, animate=True):
    steps = 0                   #holds the step count 
    ticks = 0                   #Holds the number of directional changes in short -> prevent directional change loop
    last_pos = None             #Holds the last position
    recent = []                 #list to hold recent -> prevents short loops A->B->C
    history = {}                #dictionary to keep all history -> prevents long loops

    #base case - if agent is on top of gold shouldn't happen cause of board generation
    if board.agent_pos == board.gold_pos and not board.has_gold:
        board.actuate(ww.AgentAction.grab)

    #loop until we run out of steps or you run out of directional changes
    while steps < max_steps and ticks < max_steps*10:
        ticks += 1              #prevents infinite turning 

        perception = board.perceive()     #checks tiles around agent safe if no breeze or stench
        print(f"DEBUG: facing={board.agent_facing}, stench={perception.stench}, breeze={perception.breeze}, arrows={board.num_arrows}")
        
        #if gold has already been grabbed - we win, exit loop
        if(board.has_gold):
            break
        #if glitter is true grab gold and exit loop
        if perception.glitter or (board.agent_pos == board.gold_pos and not board.has_gold):
          board.actuate(ww.AgentAction.grab)
          break

        #If the gold is in an adjacent tile, move there immediately and grab it
        r0, c0 = board.agent_pos
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = r0 + dr, c0 + dc
            if board.is_in_bounds(nr, nc) and (nr, nc) == board.gold_pos:
                desired_facing = change_facing_direction((r0, c0), (nr, nc))
                for action in turns_needed(board.agent_facing, desired_facing):
                    board.actuate(action)
                    
                    #Print board
                    if animate:
                        clear_output(wait=True)
                        ww.print_board_text(board)
                        time.sleep(sleep)

                board.actuate(ww.AgentAction.move)  #move agent
                steps += 1

                #Grab gold
                if board.agent_pos == board.gold_pos and not board.has_gold:
                    board.actuate(ww.AgentAction.grab)
                    if animate:
                        clear_output(wait=True)
                        ww.print_board_text(board)
                        time.sleep(sleep)
                break #exit loop

        suspect = set()                   #tiles with stench or breeze

        #0=forward, 1=right, 2=behind, 3=left -> relative to agent facing
        directions = [(0, +1), (+1, 0), (0, -1), (-1, 0)]
        facing = board.agent_facing.value
        r0, c0 = board.agent_pos

        has_adj_safe = False
        #check all the directions, if no breeze and no stench, neighbor is safe
        for i in range(4):
            dr, dc = directions[(facing + i) % 4]
            nr, nc = r0 + dr, c0 + dc
            if not board.is_in_bounds(nr, nc):
                continue

            if (not perception.breeze[i]) and (not perception.stench[i]):
                safe.add((nr, nc))
                has_adj_safe = True
            else:
                #if there a stench or breezed tile there, treat as risky
                suspect.add((nr, nc))

        #Shoots the Wumpus early if there is no available safe tiles at start
        if not has_adj_safe and any(perception.stench):
            if try_shoot_wumpus(board, perception, animate=animate, sleep=sleep):
                continue  #restart loop with fresh perception after the shot

        visited.add(board.agent_pos)        #Add current tile to visisted set

        #Build frontier, prefer unvisited, else allow all safe
        frontier_base = ({c for c in safe if c not in visited} or set(safe))
        frontier_base.discard(board.agent_pos)

        #Avoid last_pos and recent, but don't eliminate all options
        candidates = set(frontier_base)
        if last_pos is not None:
            candidates.discard(last_pos)
        candidates -= set(recent) if 'recent' in locals() else set()

        #if we killed all possible move by avoiding backtracks, allow backtracking
        frontier = candidates or frontier_base
        avoid = set(recent)
        if last_pos is not None:
            avoid.add(last_pos)
        alt = frontier - avoid
        if alt:
            frontier = alt  #use non-recent choice, otherwise keep original frontier
       
        #If still nothing, allow a risky (suspect) step that isn't a known hazard
        if not frontier:
            risky_frontier = {
                cell for cell in suspect
                if board.is_in_bounds(*cell) and not board.is_hazard(*cell)
            }
            #still avoid last_pos/recent if possible
            risky_candidates = risky_frontier - ({last_pos} if last_pos else set())
            risky_candidates -= set(recent) if 'recent' in locals() else set()
            frontier = risky_candidates or risky_frontier

        #If we still have no candidates, try shooting, else stop cleanly
        if not frontier:
            if any(perception.stench) and try_shoot_wumpus(board, perception, animate=animate, sleep=sleep):
                continue  #re-perceive after the shot
            print("DEBUG: No frontier; stopping.")
            break


        #Prefer unvisited cells; heavily penalize picking a visited one
        BACKTRACK_PENALTY = 100         #penalty to visited tiles prevents AI from going in circles
        def goal_cost(c):
            return Manhattan_distance(board.agent_pos, c) + (0 if c not in visited else BACKTRACK_PENALTY)

        subgoal = min(frontier, key=goal_cost)

        #plans a path add risk to tiles with stench and breeze making them higher in value
        path = a_star(board, board.agent_pos, subgoal, suspect=suspect, risk_cost = 3)

        #if no path try to loop again and break
        if not path or len(path) < 2:
            if try_shoot_wumpus(board, perception, animate=animate, sleep=sleep):
              continue
            print("DEBUG: No path; stopping.")
            break

        #Take the next step in path
        next_tile = path[1]

        #If we'd immediately backtrack and we still smell stench, shoot instead
        if last_pos is not None and next_tile == last_pos and any(perception.stench):
            if try_shoot_wumpus(board, perception, animate=animate, sleep=sleep):
                continue  #re-perceive after the shot
        
        prev_pos = board.agent_pos      #saves previous position -> to prevent loops

        #turn to face the tile we are moving into
        desired_facing = change_facing_direction(board.agent_pos, next_tile)
        for action in turns_needed(board.agent_facing, desired_facing):
            board.actuate(action)

            #prints board
            if animate:
                clear_output(wait=True)
                ww.print_board_text(board)
                time.sleep(sleep)


        board.actuate(ww.AgentAction.move)      #move forward one tile
        steps += 1                              #add to step count
        last_pos = prev_pos                     #keeps track of last pos
        recent.append(prev_pos)                 #add previous position list
        if len(recent) > 3:                     #removes a tile if 3+
          recent.pop(0)

        # Use fresh perception after moving
        p_now = board.perceive()
        
        #Tracks how many times we visit this position
        history[board.agent_pos] = history.get(board.agent_pos, 0) + 1

        #If we’ve been here too many times, break the loop by shooting or stopping
        if history[board.agent_pos] > 2:   #seen this square 3+ times
            if any(p_now.stench) and try_shoot_wumpus(board, p_now, animate=animate, sleep=sleep):
                continue
            print("DEBUG: Loop detected; stopping.")
            break

        #Grab the gold if agent is a top of it and print out the last board
        if board.agent_pos == board.gold_pos and not board.has_gold:
            board.actuate(ww.AgentAction.grab)

            if animate:
                clear_output(wait=True)
                ww.print_board_text(board)
                time.sleep(sleep)
            break

        #Shows the board after moving.
        if animate:
            clear_output(wait=True)
            ww.print_board_text(board)
            time.sleep(sleep)

    #Prints a summary of what is going on.
    frontier_count = len({cell for cell in safe if cell not in visited})
    print(f"visited={len(visited)} safe={len(safe)} frontier={frontier_count}")
    print(f"Done in {steps} steps. At {board.agent_pos} (goal={goal}).")


def main():
    #uncomment if you want to make a tiny board to test. Make a tiny board to check out algorithm 
    board = ww.generate_board(4)
    visited, safe = init_memory(board.agent_pos)
    ww.print_board_text(board)
    move_agent(board, board.gold_pos, visited, safe, sleep=0.2, animate=True)

if __name__ == "__main__":
    main()