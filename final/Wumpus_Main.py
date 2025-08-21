import Wumpus_World
import A_Star
import GA
import RL_Q_Learning


def main():
    """run the main functions of each of the different algorithms"""

print("----- Simple Reflex Agent -----")
Wumpus_World.main() 
print("------------------------\n\n")

print("----- A* Agent -----")
A_Star.main() 
print("------------------------\n\n")

print("----- Genetic Algorithm Agent -----")
GA.main() 
print("------------------------\n\n")

print("----- Reinforced Learning Agent -----")
## set save_graphs = False to show the graphs instead of saving them
## saving is better for batch runs since the tool waits for shown
## graphs to be closed before continuing
RL_Q_Learning.main() 
print("------------------------\n\n")

print ("Thank you! Goodbye.")

if __name__ == "__main__":
    main()