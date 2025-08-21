import random
import copy
from functools import reduce
import matplotlib.pyplot as plt
from Wumpus_World import *
from enum import Enum

# class Agent:
#     """base class for an agent that controls the Robot"""
    
#     def __init__(self, world: WumpusBoard):
#         self.world = world

#     def _perceive(self) -> Perception:
#         """get information about the state of the Robot in the world"""
#         return self.world.perceive()

#     ## this command should be overwritten based on type of agent
#     def _choose_action(self, view: Perception) -> AgentAction:
#         """chose the action to do based on the perception of the world and internal rules, etc."""
#         pass

#     def update(self, view_before: Perception, view_after: Perception, action: AgentAction):
#         """
#         This is a hook for a callback that has the view before performing an action,
#         the action itself, and the view after performing the action.
#         This can be used, for example, to update the Q matrix before continuing to 
#         the next action
#         """
#         pass

#     def return_val(self) -> int:
#         """
#         The value to return at the end of a run
#         In the case of GA this would be a measure of fitness
#         This is also a good place to put any relevant asserts
#         """
#         pass

#     def _actuate(self, action: AgentAction) -> None:
#         """do a command according to the rule table"""
#         self.world.actuate(action)

#     def run(self, num_steps: int) -> int:
#         """run the agent until the world is clean or until passing num_step steps
#         return the number of steps"""
#         steps = 0
#         view_before = self._perceive()
#         action = None
#         while not view_before.death and not view_before.win and steps < num_steps: 
#             steps += 1
#             action = self._choose_action(view_before)
#             self._actuate(action)
#             view_after = self._perceive()
#             self.update(view_before, view_after, action) 
#             ## TODO - remove the next line after implementing the return path
#             if view_before.glitter and action == AgentAction.grab:
#                 view_after.win = True
#             view_before = view_after
#         return self.return_val()



class GA_agent:
    """
    This is the class that includes the string representation of the wumpus board states
    and the agent's actions with regard to handling each state

    A fitness calculation is performed at init and stored for future use

    Each element in the list may undergo a random mutation of its value (column)
    according to a mutation percentage

    In order to reduce the state space, there are certain states where there should
    only be one correct action, so we will handle those cases separately
        -- if the current location glitters, always grab the gold.
        -- if there is a stench ahead of you, always shoot the arrow
           (otherwise never shoot)
    
    Also, in order to simplify the task, just focus on getting to the gold.
    A next step could be to record ("remember") our path to the gold and then
    to just walk the exact same path backwards once we have the gold.

    The fitness function itself will be computed by running simulations in random wumpus worlds and 
    recording the number of points collected according to a reward scheme.

    In addition to limiting unnecessary options, we may need to add more information in order to
    discern between diffeernt similar states (e.g. if all adjacent cells are empty, we don't want 
    to just do the same thing every time)
    """
    
    def __init__(self, world:WumpusBoard, mutation_percent: int, string_rep = None):
        self.world = world
        if mutation_percent is not None:
            self.mutation_percent = mutation_percent
        assert(self.mutation_percent is not None)
        self.string_rep = []
        self.past_actions = []
        self.win = False
        self.death = False
        ##### These need to be coordinated with the start position of the agent on the board #####
        self.visited = set(world.agent_pos) ## a list of locations that have been visited
        self.row, self.col = world.agent_pos
        self.board_size = world.n
        self.facing = world.agent_facing
        ##########################################################################################
        self.size = 2**9 ## 2**4 (options of stench) * 2**4 (options of breeze) * 2 (options of bump) * location * facing

        if string_rep is not None:
            assert len(string_rep) == self.size
            for val in string_rep:
                ## verify that all actions are legal
                # assert type(val) == int and val >= 0 and val < len(AgentAction)
                ## for now, make sure that none of the actions are chosen if they are only used for
                ## specific, predefined situations.
                assert val not in [AgentAction.grab, AgentAction.shoot ]
            self.string_rep = copy.copy(string_rep)
        else:
            for idx in range(self.size):
                self.string_rep.append(random.choice([AgentAction.move, AgentAction.turn_left, AgentAction.turn_right]))
        
        self.mutate()
        self.compute_fitness()


    def _perceive(self) -> Perception:
        """get information about the state of the Robot in the world"""
        return self.world.perceive()

    def _actuate(self, action: AgentAction) -> None:
        """do a command according to the rule table"""
        self.world.actuate(action)

    def run(self, num_steps: int) -> int:
        """run the agent until the world is clean or until passing num_step steps
        return the number of steps"""
        steps = 0
        view_before = self._perceive()
        action = None
        while not view_before.death and not view_before.win and steps < num_steps: 
            steps += 1
            action = self._choose_action(view_before)
            self._actuate(action)
            view_after = self._perceive()
            self.update(view_before, view_after, action) 
            ## TODO - remove the next line after implementing the return path
            if view_before.glitter and action == AgentAction.grab:
                view_after.win = True
            view_before = view_after
        return self.return_val()


    def compute_fitness(self) -> None:
        """test agent on board and compute average reward as fitness"""
        ## TODO - should this be done multiple times on multiple worlds?
        num_steps = 200 ## TODO - configure
        self.fitness = 0 ## initialize
        self.run(num_steps) ## self.fitness is updated during the run by the update method
    
    def _get_next_location(self, pos, direction: AgentDirection):
        row, col = pos
        match direction:
            case AgentDirection.right:
                return (row, col + 1)
            case AgentDirection.left:
                return (row, col - 1)
            case AgentDirection.down:
                return (row + 1, col)
            case AgentDirection.up:
                return (row - 1, col)
        assert(false)
        return (row,col)
    
    def _is_in_bounds(self, pos) -> bool:
        row, col = pos
        return 1 <= row <= self.board_size and 1 <= col <= self.board_size

    def _facing_wall(self, pos = None, direction: AgentDirection = None) -> bool:
        """return true if the agent is facing a wall"""
        if pos is None:
            pos = (self.row, self.col)
        if direction is None:
            direction = self.facing
        next_location = self._get_next_location(pos, direction)
        in_bounds = self._is_in_bounds(next_location)
        return not in_bounds

    def _get_wall_neighbor_list(self) -> list[bool]:
        """return a list with four boolean values indicating if there is a wall
            in front, to the right, behind, or to the left
        """
        direction = self.facing
        walls = []
        for _ in range(4):
            walls.append(self._facing_wall(pos=(self.row,self.col), direction=AgentDirection(direction)))
            direction = direction.turn_right()
        return walls
    
    def _get_visited_neighbor_list(self):
        direction = self.facing
        visited = []
        for _ in range(4):
            visited.append(self._get_next_location(pos=(self.row,self.col), direction=AgentDirection(direction)) in self.visited)
            direction = direction.turn_right()
        return visited

    def _choose_action(self, perception: Perception) -> AgentAction:
        """chose the action to do based on the perception of the world and internal rules, etc."""
        ## choose action in specific situations
        if perception.glitter:
            return AgentAction.grab
        
        if perception.stench[0]:
            assert(self.world.num_arrows > 0)
            return AgentAction.shoot
        
        if random.random() < 0.1:
            action = random.choice([AgentAction.turn_left, AgentAction.turn_right, AgentAction.move])
            assert(action in [AgentAction.turn_left, AgentAction.turn_right, AgentAction.move])
            return action

        ##choose based on the string representation    
        idx = self._get_representation(perception)
        action = self.string_rep[idx]
        assert(action in [AgentAction.turn_left, AgentAction.turn_right, AgentAction.move])
        
        if perception.bump and action == AgentAction.move:
            ## don't bump into a wall forever
            action = random.choice([AgentAction.turn_left, AgentAction.turn_right])
            assert(action in [AgentAction.turn_left, AgentAction.turn_right])
        
        # if not AgentAction.move in self.past_actions[:-4]:
        #     ## if you don't percieve anything then move to another location
        #     ## at the very least you'll eventually hit a wall...
        #     return AgentAction.move
        assert(action in [AgentAction.turn_left, AgentAction.turn_right, AgentAction.move])
        return action

    def _get_representation(self, perception: Perception) -> int:
        """get the part of the perception as represented by the index of the string representation"""
        # bool_list = [self._facing_wall()] + perception.breeze + perception.stench
        bool_list = [self._facing_wall()] + self._get_visited_neighbor_list() + perception.breeze
        ## the following was taken from @pythonwiz on reddit
        def f(a, b):
            return (a << 1) | b
        
        base_number =  reduce(f, bool_list)
        return base_number
    
    def _get_representation_action(self, perception: Perception) -> AgentAction:
        """get the action as represented by the string representation"""
        idx = self._get_representation(perception)
        ## return the action at the relevant index
        return self.string_rep[idx]

    def update(self, view_before: Perception, view_after: Perception, action: AgentAction):
        """
        update the fitness value during the run based on rewards
        """
        ## TODO - make these values configurable
        assert(action is not None)
        if view_after.death:
            self.death = True
            self.fitness -= 100
        elif view_before.glitter and action == AgentAction.grab:
            self.win = True
            self.fitness += 1000 ## found the gold. temporary win condition
        elif view_after.scream:
            self.fitness += 100 ## killed wumpus
        elif view_after.bump:
            self.fitness -= 10
        else:
            self.fitness -= 1 ## don't waste time
        self.past_actions.append(action)

        ## update our internal sense of our location and which way we are facing
        # match action:
        if action == AgentAction.move:
            if not view_after.bump:
                if self.facing == AgentDirection.right:
                    if not self.col < self.board_size:
                        print(self.row, self.col, self.visited)
                        assert(False)
                    self.col += 1
                elif self.facing == AgentDirection.left:
                    assert(self.col > 1)
                    self.col -= 1
                elif self.facing == AgentDirection.up:
                    assert(self.row > 1)
                    self.row -= 1
                elif self.facing == AgentDirection.down:
                    assert(self.row < self.board_size)
                    self.row += 1
                if (self.row, self.col) not in self.visited:
                    self.visited.add((self.row, self.col))
                    self.fitness += 10 ## small bonus for exploring a new state
        elif action == AgentAction.turn_left:
            self.facing = self.facing.turn_left()
        elif action ==  AgentAction.turn_right:
            self.facing = self.facing.turn_right()

        self.row, self.col = self.world.agent_pos
        self.facing = self.world.agent_facing

    def return_val(self) -> int:
        """
        The value to return at the end of a run
        In the case of GA this would be a measure of fitness
        This is also a good place to put any relevant asserts
        """
        perception = self._perceive()
        ## return if win, loose, or neither
        return (perception.win, perception.death)

    def print_element(self) -> None:
        """print information about the agent"""
        
        perception = self.world.perceive()
        print((self.row,self.col, self.facing), self._get_wall_neighbor_list(), self._get_visited_neighbor_list())
        perception.debug_print()
        print(f"\nFitness: {self.fitness}, past_actions: {self.past_actions[-20:]}\n")
    
    def mutate(self) -> None:
        """mutate each position according to mutate_percent"""
        for idx in range(1, self.size):
            if random.random() < (self.mutation_percent / 10000):
                self.string_rep[idx] = random.choice([AgentAction.move, AgentAction.turn_left, AgentAction.turn_right])

        
def generate_children(world: WumpusBoard, elem1: GA_agent, elem2: GA_agent):
    """Take two parent string_rep and generate two children from them
        Make sure to use the world created for this next generation
    """

    assert(elem1.size == elem2.size)
    assert(elem1.mutation_percent == elem2.mutation_percent)

    ## chose a random crossover cutpoint - not an edge
    cut_idx = random.choice(range(elem1.size))
    
    ## create the two new position strings
    child1 = elem1.string_rep[:cut_idx] + elem2.string_rep[cut_idx:]
    child2 = elem2.string_rep[:cut_idx] + elem1.string_rep[cut_idx:]
    assert(len(child1) == len(child2))

    ## return the two new children created from the
    ## string_rep. (the creation of children will 
    ## include random mutation)
    return GA_agent(world, elem1.mutation_percent, child1), GA_agent(world, elem1.mutation_percent, child2)

def choose_parents(population):
    """accepts a list of elements and chooses two parents
    based on their fitness"""

    weights = [elem.fitness for elem in population]
    min_weight = abs(min(weights))
    weights = [weight + min_weight + 1 for weight in weights]
    parents =  random.choices(population, weights, k=1)
    parents += random.choices(population, weights, k=1)

    return parents

def main(save_graphs: bool = True):
    """The main loop to collect all the runs"""
    BOARD_SIZE = 4
    MAX_ITER = 2000        ## number of iterations to run
    POPULATION_SIZE = 200  ## should be even to allow retention of original size

    MUTATION_CHANCE = 300   ## divie this by 10,000 for actual mutation chance
                            ## so that this number can be used without decimal
                            ## in plot figure name

    world = generate_board(BOARD_SIZE)
    population = [GA_agent(world, MUTATION_CHANCE) for _ in range(POPULATION_SIZE)]
    best_elem = max(population, key= lambda elem: elem.fitness)
    worst_elem = min(population, key= lambda elem: elem.fitness)
    iter = 0

    ## store initial data for plot
    best_fitnesses = [best_elem.fitness]
    ave_fitnesses = [sum([elem.fitness for elem in population]) / len(population)]

    ave_wins = [sum([elem.win for elem in population]) * 100 / len(population)]
    ave_deaths = [sum([elem.death for elem in population]) * 100 / len(population)]

    # while (best_elem.fitness < MAX_FITNESS  and iter < MAX_ITER):
    while (iter < MAX_ITER):
        iter += 1
        next_gen = []
        
        ## generate a new world on which to test the fitness
        world = generate_board(BOARD_SIZE)


        for _ in range(POPULATION_SIZE // 2):
            parents = choose_parents(population)
            assert(len(parents) == 2)
            children = generate_children(copy.copy(world), parents[0], parents[1])
            next_gen += children

        assert(len(next_gen) == POPULATION_SIZE)
        population = next_gen
        if iter % 100 == 0:
            best_elem = max(population, key= lambda elem: elem.fitness)
            worst_elem = min(population, key= lambda elem: elem.fitness)
            best_fitnesses.append(best_elem.fitness)
            ave_fitness = sum([elem.fitness for elem in population]) / len(population)
            ave_fitnesses.append(ave_fitness)
            ave_win = sum([elem.win for elem in population]) * 100 / len(population) 
            ave_wins.append(ave_win)
            ave_death =  sum([elem.death for elem in population]) * 100 / len(population) 
            ave_deaths.append(ave_death)
            print_board_text(world)
            # best_elem.print_element()
            print(f"{iter}: best={best_elem.fitness}, worst={worst_elem.fitness}, ave: {ave_fitness}")
            print(f"{iter}: wins={ave_win}%, deaths={ave_death}%")

    # ## print the best solution
    # print(f"The best element after {iter} iterations is:")
    # best_elem.print_element()
    # print(f"The worst element after {iter} iterations is:")
    # worst_elem.print_element()

    # plot(best_fitness=best_fitnesses, ave_fitness=ave_fitnesses, title=f"GA_agents_pop_{POPULATION_SIZE}_mut_{MUTATION_CHANCE}")

    iters = range(len(best_fitnesses))
    ## design fitness graph
    plt.figure(figsize=(10, 6), layout='constrained')
    plt.plot(iters, best_fitnesses, label='best fitness')  # Plot some data on the (implicit) Axes.
    plt.plot(iters, ave_fitnesses, label='average fitness')  # etc.
    plt.xlabel('runs')
    plt.ylabel('fitness')
    plt.title("Wumpus GA")
    plt.legend()
    ## save image
    if save_graphs:
        plt.savefig(f"GA_agents_fitness_pop_{POPULATION_SIZE}_mut_{MUTATION_CHANCE}.jpg")
    else:
        plt.show()

    iters = range(len(ave_wins))
    ## design win/loss graph
    plt.figure(figsize=(10, 6), layout='constrained')
    plt.plot(iters, ave_wins, label='average wins')  # Plot some data on the (implicit) Axes.
    plt.plot(iters, ave_deaths, label='average deaths')  # etc.
    plt.xlabel('runs')
    plt.ylabel('% success')
    plt.title("Wumpus GA")
    plt.legend()
    ## save image
    if save_graphs:
        plt.savefig(f"GA_agents_wins_pop_{POPULATION_SIZE}_mut_{MUTATION_CHANCE}.jpg")
    else:
        plt.show()
    return
    

if __name__ == "__main__":
    ## set to False to avoid plotting graphs
    ## graphs are cool, but stall the program until closed
    main(save_graphs=True)

