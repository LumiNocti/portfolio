# 8/14/2025
"""Fixes Done so far : 
1. Removed duplicate functions - load_images() and generate_board() were defined twice
2. Fixed climb logic - Now checks self.has_gold instead of self.gold_pos is None
3. Added has_gold tracking - New boolean to track if agent grabbed the gold
4. Fixed perception timing - Glitter only shows when standing on ungrabbed gold
5. Improved stench logic - Only shows stench if wumpus is alive
6. Added bounds checking for directional sensors
7. Enhanced debug output - Shows gold status and arrow count

Added simple agent implementation : 

1. Agent Auto-grabs gold when standing on it 
2. Auto-shoots wumpus when facing it with arrows available 
3. Basic pathfinding back to start when carrying gold 
4. Simple danger avoidance (turns away from breeze/stench) 

The agent should now work much better for testing your genetic algorithm. 
The main loop structure implements the run_simple_agent() function.
"""

# Wumpus World Board Generator and Renderer [Google Colab Version] 
# This module generates random Wumpus World game board with Validation and saves output as image 

import random
import pygame
import time
import os
from collections import deque
from typing import List, Tuple, Optional
from enum import Enum

# Local Image Path
AGENT_IMG = "Images/Agent-Icon.svg"
WUMPUS_IMG = "Images/Wumpus-Icon.svg"
PIT_IMG = "Images/DarkPit-Icon.svg"
GOLD_IMG = "Images/Gold-Icon.svg"
BREEZE_IMG = "Images/Breeze-Icon.svg"
STENCH_IMG = "Images/Stench-Icon.svg"
GLITTER_IMG = "Images/Glitter-Icon.svg"

# Global Image Cache
IMAGE_CACHE = {}

class AgentDirection(Enum):
    # Indicates if the agent is facing up (north), down (south), left (west), or right (east)
    right = 0
    down = 1
    left = 2
    up = 3 

    def turn_right(self):
        return AgentDirection((self.value + 1) % len(AgentDirection))

    def turn_left(self):
        return AgentDirection((self.value - 1) % len(AgentDirection))

class AgentAction(Enum):
    # These are the possible actions that the agent can actuate 
    move = 0        # move forward one cell
    turn_right = 1  # turn 90 degrees clockwise
    turn_left = 2   # turn 90 degrees counter clockwise
    grab = 3        # grab the gold (in the location with the gold)
    shoot = 4       # shoot the wumpus that is in the cell directly in front of you
    # climb = 5       # after reaching (1,1) with the gold, climb up the ladder to safety

class Perception():
    # These is the information the agent can get from the environment 
    def __init__(self, death=False, win=False, bump=False, scream=False, glitter=False, breeze=None, stench=None):
        self.death = death      # (bool) The agent died and the game is over 
        self.win = win          # (bool) you safely climbed out of the maze with the gold
        self.bump = bump        # (bool) will be true after a move action where the agent bumped into a wall
        self.scream = scream    # (bool) will be true after a shoot action that kills the wumpus
        self.glitter = glitter  # (bool) indicates that you are in the goal cell and have not yet picked up the gold
        self.breeze = breeze    # (bool, bool, bool, bool) indicates if there is a pit to your
                               # (forward, right, behind, left) or None indicates none
        self.stench = stench    # (bool, bool, bool, bool) indicates if there is a live wumpus to your
                               # (forward, right, behind, left) or None indicates none
    
    def debug_print(self):
        print(f"death={self.death}, win={self.win}, bump={self.bump}, scream={self.scream}, glitter={self.glitter}")
        print(f"breeze={self.breeze}, stench={self.stench}")

class WumpusBoard:
    # Represents a Wumpus World board with all game elements and sensors

    def __init__(self, n: int):
        self.n = n
        self.initial_agent_pos = (1,1) # Agent starts at (1,1) - using 1-indexed
        self.agent_pos = self.initial_agent_pos   
        self.agent_facing = AgentDirection.right
        self.wumpus_pos = None
        self.gold_pos = None
        self.pit_positions = set()
        self.sensors = {}  # (row, col) -> list of sensor types
        self.perception = Perception()  # default of nothing
        self.num_arrows = 1  # start with one chance to shoot the wumpus
        self.has_gold = False  # track if agent has grabbed the gold

    def perceive(self) -> Perception:
        # Perceive the neighborhood of the agent 
        # Check for death first
        if self.is_hazard(self.agent_pos[0], self.agent_pos[1]):
            self.perception.death = True
            return self.perception  # quick exit upon death
        
        # Check for glitter (only if standing on gold and haven't grabbed it)
        if self.agent_pos == self.gold_pos and not self.has_gold:
            self.perception.glitter = True
        else:
            self.perception.glitter = False
        
        # Check directional sensors
        breeze = []
        stench = []
        direction = self.agent_facing
        
        for idx in range(len(AgentDirection)):
            next_pos = self.get_next_position(direction)
            # Only add stench if wumpus is alive and in that direction
            stench.append(self.wumpus_pos == next_pos and self.wumpus_pos is not None)
            # Check if there's a pit in that direction
            breeze.append(next_pos in self.pit_positions if self.is_in_bounds(*next_pos) else False)
            direction = direction.turn_right()
        
        self.perception.breeze = breeze
        self.perception.stench = stench
        return self.perception

    def actuate(self, action: AgentAction):
        # Implement the effect of the agent's action on the board
        # Reset perception but keep death and win state
        old_death = self.perception.death
        old_win = self.perception.win
        self.perception = Perception()
        
        # If already dead or won, no actions allowed
        if old_death or old_win:
            self.perception.death = old_death
            self.perception.win = old_win
            return
        
        match(action):
            case AgentAction.turn_right:
                self.agent_facing = self.agent_facing.turn_right()
            
            case AgentAction.turn_left:
                self.agent_facing = self.agent_facing.turn_left()
            
            case AgentAction.move:
                next_pos = self.get_next_position()
                if self.is_in_bounds(next_pos[0], next_pos[1]):
                    self.agent_pos = next_pos
                else:
                    self.perception.bump = True
            
            case AgentAction.grab:
                if self.gold_pos == self.agent_pos and not self.has_gold:
                    self.has_gold = True
                    # Remove gold from board since it's been grabbed
                    self.gold_pos = None
                    self.perception.win = True
            
            case AgentAction.shoot:
                if self.num_arrows > 0:
                    self.num_arrows -= 1
                    target_pos = self.get_next_position()
                    if target_pos == self.wumpus_pos:
                        self.perception.scream = True 
                        self.wumpus_pos = None  # wumpus is dead
            
            # case AgentAction.climb:
            #     if self.agent_pos == self.initial_agent_pos and self.has_gold:
            #         self.perception.win = True

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        # Get valid orthogonal neighbors (4-directional)
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 1 <= nr <= self.n and 1 <= nc <= self.n:
                neighbors.append((nr, nc))
        return neighbors

    def get_next_position(self, direction: AgentDirection = None) -> Tuple[int, int]:
        deltas = [(0,1), (1,0), (0,-1), (-1,0)]  # right, down, left, up
        if direction is None:
            direction = self.agent_facing
        pr, pc = self.agent_pos
        dr, dc = deltas[direction.value]
        return (pr + dr, pc + dc)

    def is_in_bounds(self, row: int, col: int) -> bool:
        return 1 <= row <= self.n and 1 <= col <= self.n

    def is_hazard(self, row: int, col: int) -> bool:
        # Check if position contains a hazard (wumpus or pit).
        return (row, col) == self.wumpus_pos or (row, col) in self.pit_positions

    def is_not_pit(self, row: int, col: int) -> bool:
        # This indicates that the agent can potentially step here
        assert(1 <= row <= self.n and 1 <= col <= self.n)
        return not (row, col) in self.pit_positions

    def is_safe(self, row: int, col: int) -> bool:
        # This indicates that the agent can enter this position without dying 
        assert(self.is_in_bounds(row, col))
        return not self.is_hazard(row, col)

def setup_colab_display():
    # Setup virtual display for Google Colab
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame", "pyvirtualdisplay"])
        subprocess.check_call(["apt-get", "update"])
        subprocess.check_call(["apt-get", "install", "-y", "xvfb"])
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1200, 800))
        display.start()
        print("Virtual display setup complete")
    except Exception as e:
        print(f"Display setup failed: {e}")
        print("Continuing without virtual display...")

def setup_pygame_colab():
    # Setup pygame for Google Colab environment.
    try:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        print("Pygame initialized for headless mode")
    except Exception as e:
        print(f"Pygame setup error: {e}")

def load_images():
    # Load all required images from local files with fallbacks.
    image_files = {
        'agent': AGENT_IMG,
        'wumpus': WUMPUS_IMG,
        'pit': PIT_IMG,
        'gold': GOLD_IMG,
        'breeze': BREEZE_IMG,
        'stench': STENCH_IMG,
        'glitter': GLITTER_IMG
    }

    fallback_colors = {
        'agent': (0, 255, 0),      # Green
        'wumpus': (255, 0, 0),     # Red
        'pit': (0, 0, 0),          # Black
        'gold': (255, 215, 0),     # Gold
        'breeze': (173, 216, 230), # Light blue
        'stench': (255, 165, 0),   # Orange
        'glitter': (255, 255, 0)   # Yellow
    }

    for name, filepath in image_files.items():
        try:
            if os.path.exists(filepath):
                print(f"Loading {name} from {filepath}")
                try:
                    img = pygame.image.load(filepath)
                    IMAGE_CACHE[name] = img
                    print(f"✓ {name} loaded successfully")
                except pygame.error as e:
                    print(f"Pygame can't load {name} (likely SVG): {e}")
                    img = pygame.Surface((64, 64))
                    img.fill(fallback_colors[name])
                    IMAGE_CACHE[name] = img
                    print(f"✓ {name} using colored fallback")
            else:
                print(f"File not found: {filepath}, using colored rectangle")
                img = pygame.Surface((64, 64))
                img.fill(fallback_colors[name])
                IMAGE_CACHE[name] = img
        except Exception as e:
            print(f"Error loading {name}: {e}")
            img = pygame.Surface((64, 64))
            img.fill(fallback_colors.get(name, (128, 128, 128)))
            IMAGE_CACHE[name] = img

def generate_board(n: Optional[int] = None, seed: Optional[int] = None, verbose: bool = False) -> WumpusBoard:
    # Generate a random Wumpus World board 
    if n is None or not isinstance(n, int) or n < 3:
        n = 4

    if seed is not None:
        random.seed(seed)

    max_attempts = 1000
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        board = WumpusBoard(n)

        all_positions = [(r, c) for r in range(1, n+1) for c in range(1, n+1)]
        available_positions = [pos for pos in all_positions if pos != board.agent_pos]

        if len(available_positions) < 5:
            continue

        selected = random.sample(available_positions, 5)
        board.wumpus_pos = selected[0]
        board.gold_pos = selected[1]
        board.pit_positions = set(selected[2:5])

        compute_sensors(board)

        if validate_board(board):
            if verbose:
                print(f"Valid board generated after {attempt} attempts")
            return board

    raise RuntimeError(f"Failed to generate valid board after {max_attempts} attempts")

def compute_sensors(board: WumpusBoard):
    # Compute sensor overlays for all cells.
    board.sensors = {}

    for row in range(1, board.n + 1):
        for col in range(1, board.n + 1):
            sensors = []

            if (row, col) == board.gold_pos:
                sensors.append('glitter')

            for nr, nc in board.get_neighbors(row, col):
                if (nr, nc) in board.pit_positions:
                    sensors.append('breeze')
                    break

            for nr, nc in board.get_neighbors(row, col):
                if (nr, nc) == board.wumpus_pos:
                    sensors.append('stench')
                    break

            board.sensors[(row, col)] = sensors

def validate_board(board: WumpusBoard) -> bool:
    # Validate board meets all requirements 
    agent_neighbors = board.get_neighbors(*board.agent_pos)
    safe_moves = [pos for pos in agent_neighbors if board.is_not_pit(*pos)]

    if not safe_moves:
        return False

    if not has_path(board, board.agent_pos, board.gold_pos):
        return False

    if not has_path(board, board.gold_pos, board.agent_pos):
        return False

    return True

def has_path(board: WumpusBoard, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    # Check if path exists between two positions using BFS 
    if start == goal:
        return True

    queue = deque([start])
    visited = {start}

    while queue:
        current = queue.popleft()
        for neighbor in board.get_neighbors(*current):
            if neighbor == goal:
                return True
            if neighbor not in visited and board.is_not_pit(*neighbor):
                visited.add(neighbor)
                queue.append(neighbor)

    return False

def render_board_to_file(board: WumpusBoard, filename: str = "wumpus_board.png", cell_size: int = 80):
    # Render the board and save as image file 
    if not IMAGE_CACHE:
        load_images()

    pygame.init()
    window_size = board.n * cell_size
    screen = pygame.Surface((window_size, window_size))

    WHITE = (255, 255, 255)
    GRID_COLOR = (200, 200, 200)
    screen.fill(WHITE)

    for row in range(1, board.n + 1):
        for col in range(1, board.n + 1):
            x = (col - 1) * cell_size
            y = (row - 1) * cell_size

            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(screen, GRID_COLOR, rect, 2)

            if (row, col) == board.agent_pos:
                draw_scaled_image(screen, 'agent', x, y, cell_size)
            elif (row, col) == board.wumpus_pos:
                draw_scaled_image(screen, 'wumpus', x, y, cell_size)
            elif (row, col) in board.pit_positions:
                draw_scaled_image(screen, 'pit', x, y, cell_size)
            elif (row, col) == board.gold_pos:
                draw_scaled_image(screen, 'gold', x, y, cell_size)

            sensors = board.sensors.get((row, col), [])
            for i, sensor in enumerate(sensors):
                sensor_size = cell_size // 4
                offset_x = (i % 2) * (cell_size - sensor_size)
                offset_y = (i // 2) * (cell_size - sensor_size)
                draw_scaled_image(screen, sensor, x + offset_x, y + offset_y, sensor_size)

    pygame.image.save(screen, filename)
    print(f"Board saved as {filename}")
    pygame.quit()

def draw_scaled_image(screen, image_name: str, x: int, y: int, size: int):
    # Draw scaled image at specified position.
    if image_name in IMAGE_CACHE:
        img = pygame.transform.scale(IMAGE_CACHE[image_name], (size, size))
        screen.blit(img, (x, y))

def display_board_in_colab(filename: str = "wumpus_board.png"):
    # Display the generated board image in Colab 
    try:
        from IPython.display import Image, display
        display(Image(filename))
    except ImportError:
        print(f"Board saved as {filename} - download to view")

def print_board_text(board: WumpusBoard):
    # Print text representation of the board for debugging.
    print(f"\n{board.n}x{board.n} Wumpus World Board:")
    print("=" * (board.n * 4 + 1))

    for row in range(1, board.n + 1):
        row_str = "|"
        for col in range(1, board.n + 1):
            if (row, col) == board.agent_pos:
                cell = " A "
            elif (row, col) == board.wumpus_pos:
                cell = " W "
            elif (row, col) in board.pit_positions:
                cell = " P "
            elif (row, col) == board.gold_pos:
                cell = " G "
            else:
                cell = " . "
            row_str += cell + "|"
        print(row_str)
        print("-" * (board.n * 4 + 1))

    print("\nLegend: A=Agent, W=Wumpus, G=Gold, P=Pit, .=Empty")
    print(f"Agent at: {board.agent_pos} facing {board.agent_facing.name}")
    print(f"Wumpus at: {board.wumpus_pos}")
    print(f"Gold at: {board.gold_pos}")
    print(f"Pits at: {list(board.pit_positions)}")
    print(f"Has gold: {board.has_gold}")
    print(f"Arrows remaining: {board.num_arrows}")

def create_wumpus_world(n: int = 4, seed: Optional[int] = None, save_image: bool = False, verbose: bool = False, colab: bool = False) -> WumpusBoard:
    # Main Function to create and display Wumpus World board in Colab 
    if colab:
        setup_pygame_colab()
        setup_colab_display()
    board = generate_board(n, seed, verbose)
    if verbose:
        print_board_text(board)

    if save_image:
        render_board_to_file(board, f"wumpusworld_init_state_{n}x{n}.png")
        display_board_in_colab(f"wumpusworld_init_state_{n}x{n}.png")

    return board

def run_simple_agent(board: WumpusBoard, max_steps: int = 200) -> dict:
    """
    Run a simple agent that implements the logic you described:
    - Grab gold when standing on it
    - Shoot wumpus when facing it
    - Otherwise make basic movement decisions
    """
    perception = board.perceive()
    steps = 0
    score = 0
    
    while not perception.death and not perception.win and steps < max_steps:
        steps += 1
        
        # Auto-grab gold if standing on it
        if perception.glitter and not board.has_gold:
            board.actuate(AgentAction.grab)
            score += 1000  # Big reward for getting gold
            print(f"Step {steps}: Grabbed gold!")
            assert(board.perception.win)
        
        # Auto-shoot wumpus if facing it (and have arrows)
        elif perception.stench and perception.stench[0] and board.num_arrows > 0:  # stench[0] is forward
            board.actuate(AgentAction.shoot)
            if board.perception.scream:
                score += 100  # Reward for killing wumpus
                print(f"Step {steps}: Killed wumpus!")
        
        # If have gold, try to get back to start
        elif board.has_gold:
            assert(perception.win)
            ## TODO - implement the return to initial location with the gold
            # if board.agent_pos == board.initial_agent_pos:
            #     board.actuate(AgentAction.climb)
            #     score += 2000  # Big reward for winning
            #     print(f"Step {steps}: Won the game!")
            # else:
            #     # Simple pathfinding back to start - just head towards (1,1)
            #     current_row, current_col = board.agent_pos
                
            #     if current_row > 1:  # Need to go up
            #         if board.agent_facing != AgentDirection.up:
            #             board.actuate(AgentAction.turn_left if board.agent_facing.value > AgentDirection.up.value else AgentAction.turn_right)
            #         else:
            #             board.actuate(AgentAction.move)
            #     elif current_col > 1:  # Need to go left
            #         if board.agent_facing != AgentDirection.left:
            #             board.actuate(AgentAction.turn_left if board.agent_facing.value > AgentDirection.left.value else AgentAction.turn_right)
            #         else:
            #             board.actuate(AgentAction.move)
        
        # Exploration phase - avoid obvious dangers
        else:
            # If there's a breeze or stench ahead, turn
            if perception.breeze and perception.breeze[0] or perception.stench and perception.stench[0]:
                board.actuate(AgentAction.turn_right)
            # If just bumped, turn around
            elif perception.bump:
                board.actuate(AgentAction.turn_right)
                score -= 10  # Small penalty for bumping
            # Otherwise, try to move forward
            else:
                board.actuate(AgentAction.move)
        
        # Get new perception
        perception = board.perceive()
        
        if perception.death:
            score -= 1000  # Big penalty for dying
            print(f"Step {steps}: Agent died!")
        elif perception.win:
            print(f"Step {steps}: Agent won!")
    
    return {
        'steps': steps,
        'score': score,
        'won': perception.win,
        'died': perception.death,
        'has_gold': board.has_gold
    }

def main():
    """
    The local main function to be run when running this file directly
    and also to be called from the integrated main file
    """
    # Create a test board
    # board = create_wumpus_world(n=4, seed=42, save_image=False)
    board = create_wumpus_world(n=5, seed=int(time.time()), save_image=True, colab=True, verbose=True)
    
    # Test basic functionality
    print("\n=== Testing Basic Functionality ===")
    print(f"Initial facing: {board.agent_facing}")
    
    board.actuate(AgentAction.turn_right)
    print(f"After turn_right: {board.agent_facing}")
    
    board.actuate(AgentAction.turn_left)
    print(f"After turn_left: {board.agent_facing}")
    
    perception = board.perceive()
    perception.debug_print()
    
    # Test movement
    board.actuate(AgentAction.move)
    perception = board.perceive()
    perception.debug_print()
    print_board_text(board)
    
    # Run simple agent
    print("\n=== Running Simple Agent ===")
    fresh_board = create_wumpus_world(n=5, seed=int(time.time()), save_image=False, colab=True, verbose=True)
    results = run_simple_agent(fresh_board, max_steps=50)
    print(f"Agent Results: {results}")

# Test the fixes
if __name__ == "__main__":
    main()