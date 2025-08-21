import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

# Env API
import Wumpus_World as Wumpus_board


# =========================
# Configurable Reward Scales
# =========================
class RELRewards:
    def __init__(self):
        # Terminal events
        self.pickup_treasure = 1500
        self.died            = -800        # used if cause-specific is disabled
        self.died_to_wumpus  = -800        # used if cause-specific is enabled
        self.died_to_pit     = -900

        # Step / action costs
        self.move            = -3
        self.turning_right   = -1.0
        self.turning_left    = -1.0
        self.hit_wall        = -60

        # Shooting
        self.arrow_hit       = 200
        self.arrow_miss      = -40

        # Grabbing when no gold (we’ll never use grab, but keep for completeness)
        self.failed_pickup_treasure = -40

        # Shaping
        self.new_cell_bonus      = +4
        self.revisit_penalty     = -1
        self.forward_breeze_pen  = -8
        self.forward_stench_pen  = -12
        self.safe_forward_bonus  = +1

R = RELRewards()

# =========================
# Training Flags (easy knobs)
# =========================
CONFIG_CAUSE_SPECIFIC_DEATH = True     # penalize pit vs wumpus differently
# We WON'T force grab; we auto-win on glitter in the reward step instead


# =========================
# Helpers
# =========================
def bits4(flags) -> int:
    """Pack [forward, right, back, left] -> 4-bit int (F=8, R=4, B=2, L=1)."""
    if not flags:
        return 0
    total = 0
    if flags[0]: total += 8   # forward
    if flags[1]: total += 4   # right
    if flags[2]: total += 2   # back
    if flags[3]: total += 1   # left
    return total


def state_id(agent_pos: Tuple[int, int], facing: int, n: int,
             breeze_bits: int, stench_bits: int) -> int:
    """Pack (pos, facing, breeze_bits, stench_bits) into ONE row index."""
    r, c = agent_pos
    r0, c0 = r - 1, c - 1  # 0-based for indexing math
    return (((r0 * n + c0) * 4 + facing) * 16 + breeze_bits) * 16 + stench_bits


def argmax_tiebreak(row: np.ndarray) -> int:
    """Argmax with random tie-break."""
    m = np.max(row)
    idxs = np.flatnonzero(row == m)
    return int(np.random.choice(idxs))


# =========================
# Q-Learning (no env edits)
# =========================
def Q_Learning(learning_rate: float,
               discount: float,
               epsilon: float,
               episodes: int,
               n: int,
               max_moves: int,
               eps_min: float = 0.05,
               eps_decay: Optional[float] = None,
               log_every: int = 1000):
    """
    Generalized training: new random board every episode; facing-based percepts.
    Implements:
      - Auto-win on reaching gold (no 'grab' needed), in reward section.
      - 'Shoot any adjacent' via action-mask micro-policy that reorients then shoots.
    Returns (Q_Table, stats, history).
    """
    num_states  = n * n * 4 * 16 * 16
    num_actions = len(Wumpus_board.AgentAction)
    Q_Table     = np.zeros((num_states, num_actions), dtype=np.float32)

    # Compute decay to reach eps_min ~70% through training (if not provided)
    if eps_decay is None:
        ep_target = max(1, int(episodes * 0.7))
        eps_decay = (eps_min / max(1e-9, epsilon)) ** (1.0 / ep_target)

    wins = 0
    deaths = 0
    win_history = []
    death_history = []
    winrate_hist = []

    for ep in range(episodes):
        board = Wumpus_board.create_wumpus_world(n, seed=None, save_image=False)
        visited_cells = {board.agent_pos}

        p = board.perceive()
        moves = 0
        done = False

        while not done and moves < max_moves:
            # ----- state -----
            s = state_id(board.agent_pos, board.agent_facing.value, n,
                         bits4(p.breeze), bits4(p.stench))

            # ----- ACTION MASK (stench-aware micro-policy) -----
            mask = np.ones(num_actions, dtype=bool)

            # Never use GRAB: we auto-win on glitter in reward step
            mask[Wumpus_board.AgentAction.grab.value] = False

            # Where is the Wumpus relative to me?
            stench_dir = None  # 0=fwd, 1=right, 2=back, 3=left
            if p.stench:
                for i, v in enumerate(p.stench):
                    if v:
                        stench_dir = i
                        break

            # If we have an arrow and smell the Wumpus somewhere,
            # restrict actions to reorient toward it or shoot if forward.
            if getattr(board, "num_arrows", 0) > 0 and stench_dir is not None:
                mask[:] = False
                if stench_dir == 0:
                    mask[Wumpus_board.AgentAction.shoot.value] = True  # already ahead
                elif stench_dir == 1:
                    mask[Wumpus_board.AgentAction.turn_right.value] = True
                elif stench_dir == 3:
                    mask[Wumpus_board.AgentAction.turn_left.value] = True
                else:  # stench behind -> choose a consistent turn, e.g., right
                    mask[Wumpus_board.AgentAction.turn_right.value] = True
            else:
                # No useful shoot: disable if no arrow or no adjacency in ANY direction
                if getattr(board, "num_arrows", 0) <= 0 or not (p.stench and any(p.stench)):
                    mask[Wumpus_board.AgentAction.shoot.value] = False

                # Block MOVE if forward is a wall (avoid guaranteed bump)
                fwd_r, fwd_c = board.get_next_position()
                if not board.is_in_bounds(fwd_r, fwd_c):
                    mask[Wumpus_board.AgentAction.move.value] = False

            # ε-greedy over VALID actions
            if np.random.rand() < epsilon:
                valid = np.flatnonzero(mask)
                a_id = int(np.random.choice(valid))
            else:
                row = Q_Table[s].copy()
                row[~mask] = -1e9
                a_id = argmax_tiebreak(row)

            a = Wumpus_board.AgentAction(a_id)

            # ----- step env -----
            board.actuate(a)
            p2 = board.perceive()

            s2 = state_id(board.agent_pos, board.agent_facing.value, n,
                          bits4(p2.breeze), bits4(p2.stench))

            # ----- reward shaping -----
            rwd = 0.0

            # AUTO-WIN on landing on gold (no grab needed)
            if not done and (p2.glitter is True):
                rwd = R.pickup_treasure
                wins += 1
                done = True

            elif p2.death:
                if CONFIG_CAUSE_SPECIFIC_DEATH:
                    died_pit    = (board.agent_pos in board.pit_positions)
                    died_wumpus = (board.wumpus_pos is not None and board.agent_pos == board.wumpus_pos)
                    if died_pit:
                        rwd = R.died_to_pit
                    elif died_wumpus:
                        rwd = R.died_to_wumpus
                    else:
                        rwd = R.died
                else:
                    rwd = R.died
                done = True
                deaths += 1

            elif p2.win:
                # In case env ever flags win separately
                rwd = R.pickup_treasure
                wins += 1
                done = True

            elif a == Wumpus_board.AgentAction.move:
                rwd = R.move
                # risk-aware facing penalties (PRE-action)
                if p.breeze and p.breeze[0]:
                    rwd += R.forward_breeze_pen
                if p.stench and p.stench[0]:
                    rwd += R.forward_stench_pen

                if p2.bump:
                    rwd += R.hit_wall
                else:
                    # exploration shaping
                    if board.agent_pos not in visited_cells:
                        rwd += R.new_cell_bonus
                        visited_cells.add(board.agent_pos)
                    else:
                        rwd += R.revisit_penalty

                    # tiny bonus if forward looked safe PRE-action
                    if not (p.breeze and p.breeze[0]) and not (p.stench and p.stench[0]):
                        rwd += R.safe_forward_bonus

            elif a == Wumpus_board.AgentAction.turn_right:
                rwd = R.turning_right

            elif a == Wumpus_board.AgentAction.turn_left:
                rwd = R.turning_left

            elif a == Wumpus_board.AgentAction.shoot:
                rwd = R.arrow_hit if p2.scream else R.arrow_miss

            # ----- Q update -----
            target = rwd if done else (rwd + discount * np.max(Q_Table[s2]))
            Q_Table[s, a_id] = (1 - learning_rate) * Q_Table[s, a_id] + learning_rate * target

            moves += 1
            p = p2  # advance percept

        # epsilon decay per episode
        epsilon = max(eps_min, epsilon * eps_decay)

        # logs
        win_history.append(wins)
        death_history.append(deaths)
        winrate_hist.append(wins / (ep + 1))

        if log_every and ((ep + 1) % log_every == 0):
            print(f"Episode {ep+1}/{episodes}, epsilon={epsilon:.3f}, wins={wins}, deaths={deaths}")

    history = {"wins": win_history, "deaths": death_history, "winrate": winrate_hist}
    stats   = {"wins": wins, "deaths": deaths, "episodes": episodes}
    print(f"Training complete - total wins: {wins}/{episodes}")
    return Q_Table, stats, history


def main(save_graphs: bool = True):
    episodes  = 200_000
    learning_lr = 0.10
    gamma       = 0.97
    epsilon0    = 1.0
    eps_min     = 0.05
    n           = 4
    max_moves   = 150

    ## Train the model
    Q, stats, history = Q_Learning(
        learning_rate=learning_lr,
        discount=gamma,
        epsilon=epsilon0,
        episodes=episodes,
        n=n,
        max_moves=max_moves,
        eps_min=eps_min,
        eps_decay=None,      # auto-computed to reach eps_min ~70% into training
        log_every=1000
    )
    print("TRAIN stats:", stats)

    # ----- Plots -----
    x = np.arange(1, episodes + 1)

    # (A) Cumulative wins & deaths vs episodes
    plt.figure()
    plt.plot(x, history["wins"], label="Cumulative wins")
    plt.plot(x, history["deaths"], label="Cumulative deaths")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.title("Wumpus Q-learning: cumulative wins/deaths")
    plt.legend()
    plt.tight_layout()
    if save_graphs:
        plt.savefig("Wumpus_Q_learning_cumulative.jpg")
    else:
        plt.show()

    # (B) Win rate vs episodes
    plt.figure()
    plt.plot(x, history["winrate"], label="Average win rate")
    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.title("Wumpus Q-learning: average win rate")
    plt.legend()
    plt.tight_layout()
    if save_graphs:
        plt.savefig("Wumpus_Q_learning_average.jpg")
    else:
        plt.show()


if __name__ == "__main__":
    ## set to False to avoid plotting graphs
    ## graphs are cool, but stall the program until closed
    save_graphs = True 
    main(save_graphs=save_graphs)
