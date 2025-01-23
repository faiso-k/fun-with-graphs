import functools
import itertools
import math

from networkx import Graph, shortest_path

# choose optimization: global variables
DEV_VERBOSE = False
NO_HEURISTICS = False


@functools.cache
def evaluate(graph: Graph, cops_positions: tuple[int], robber_position: int, rounds_played: int,
             max_rounds: int) -> float | None:
    """Evaluates the current state of the game.

    :return: Positive if the cops win, negative if the robber wins. None if the game is not finished.
    """
    # TODO: change penalty for cops lost such that the longest path is chosen, to increase game moves in hopes of imperfect decisions

    possible_neighbours = ({robber_position} | set(graph.neighbors(robber_position))) - set(cops_positions)

    if possible_neighbours == set():
        return 1 / (rounds_played + 1)
    elif rounds_played > max_rounds:
        return -rounds_played - 1
    return None


def find_best_move_cops(graph: Graph, max_rounds: int, cops_count: int, cops_positions: tuple[int],
                        robber_position: int, x, y) -> list[int]:
    return list(minimax_cop(graph, 0, cops_positions, robber_position, max_rounds, cops_count)[1])


def find_best_move_robber(graph: Graph, max_rounds: int, cops_count: int, old_cop_positions, cops_positions: tuple[int],
                          robber_position: int, x, y) -> int:
    return minimax_rob(graph, 0, old_cop_positions, cops_positions, robber_position, max_rounds, cops_count)[1]


@functools.cache
def minimax_cop(graph: Graph, depth: int, cops_positions: tuple[int], robber_position: int, max_rounds: int,
                cops_count: int):
    """ Calculates the best move for the cop by considering all possible (and logical) moves using a brute forcing method.

    :param graph:
    :param depth: Recursion depth.
    :param cops_positions: Current position of the cops.
    :param robber_position: Current position of the robber.
    :param max_rounds: Maximum number of rounds.
    :param cops_count: Amount of cops.
    :return: Best possible score and according move.
    """
    score = evaluate(graph, cops_positions, robber_position, depth, max_rounds)

    if DEV_VERBOSE:
        print(" : " * depth + "c:" + str(cops_positions) + " - " + str(robber_position) + " - " + str(score))
    if score is not None:
        return [score, cops_positions]

    best_score = -math.inf
    best_move = None

    # reduce search space?
    # possible_cops_positions = itertools.combinations(reachable_graph_plus_cops(graph, current_cop_positions, robber_position), cops_count)
    # possible_cops_positions = itertools.combinations(graph.nodes, cops_count)
    possible_cops_positions = relevant_cop_moves(graph, cops_positions, robber_position, cops_count)

    for current_cop_positions in possible_cops_positions:

        score = minimax_rob(graph, depth + 1, cops_positions, tuple(current_cop_positions), robber_position, max_rounds,
                            cops_count)[0]
        if score > best_score:
            best_score = score
            best_move = current_cop_positions

    return [best_score, best_move]


@functools.cache
def minimax_rob(graph: Graph, depth: int, old_cop_position, new_cops_positions: tuple[int], robber_position: int,
                max_rounds: int, cops_count: int) -> list[float | int]:
    """ Calculates the best move for the robber by considering all possible moves using a brute forcing method.

    :param graph:
    :param depth: Recursion depth.
    :param old_cop_position: Old position of the cops.
    :param new_cops_positions: Current position of the cops.
    :param robber_position: Current position of the robber.
    :param max_rounds: Maximum number of rounds.
    :param cops_count: Amount of cops.
    :return: Best possible score and according move.
    """
    score = evaluate(graph, new_cops_positions, robber_position, depth, max_rounds)

    if DEV_VERBOSE:
        print(" : " * depth + "r:" + str(new_cops_positions) + " - " + str(robber_position) + " - " + str(score))

    if score is not None:
        return [score, robber_position]

    best_score = math.inf
    best_move = None

    reachable_nodes = generate_possible_moves_robber(graph, old_cop_position, new_cops_positions, robber_position)

    for current_robber_position in reachable_nodes:
        score = minimax_cop(graph, depth + 1, new_cops_positions, current_robber_position, max_rounds, cops_count)[0]
        if score < best_score:
            best_score = score
            best_move = current_robber_position
    return [best_score, best_move]


@functools.cache
def generate_possible_moves_robber(graph: Graph, old_cop_positions: tuple[int], cops_positions: tuple[int],
                                   robber_position: int) -> set[int]:
    controlled_nodes = set(cops_positions) & set(old_cop_positions)

    # Exclude controlled nodes to create a subgraph
    # Time complexity: O(n) where n is the number of nodes in the graph
    allowed_nodes = set(graph.nodes) - controlled_nodes

    # Use BFS to find all reachable nodes from the robber's position in the allowed subgraph
    visited = set()
    queue = [robber_position]

    # Time complexity: O(n + m) where n is the number of nodes in the graph and m is the number of edges
    while queue:
        node = queue.pop(0)
        if node in visited or node not in allowed_nodes:
            continue

        visited.add(node)
        queue.extend(neighbor for neighbor in graph.neighbors(node) if neighbor not in visited)

    # Exclude nodes currently occupied by cops
    reachable_nodes = visited - set(cops_positions)

    return reachable_nodes


@functools.cache
def relevant_cop_moves(graph: Graph, cops_positions: tuple[int], robber_position: int, cops_number: int) -> list[
    tuple[int, ...]]:
    copless_graph = graph.subgraph(set(graph) - (set(cops_positions) - {robber_position}))
    reachable_nodes = set(graph.subgraph(shortest_path(copless_graph, robber_position)).nodes) - set(cops_positions)

    considered_cops = set(cops_positions) | reachable_nodes

    cop_combinations = itertools.combinations(considered_cops, cops_number)

    return list(cop_combinations)
