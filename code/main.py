import math
import random
import time

import networkx as nx

import final_minimax
import heuristics


def create_random_connected_graph(n, e):
    """
    Creates a random connected graph with n nodes and e edges.

    Parameters:
    n (int): Number of nodes in the graph.
    e (int): Number of edges in the graph.

    Returns:
    G (networkx.Graph): A randomly generated connected graph with n nodes and e edges.
    """
    # Create an empty graph
    G = nx.Graph()

    # Add n nodes to the graph
    G.add_nodes_from(range(n))

    # Ensure the number of edges does not exceed the maximum possible for n nodes
    max_edges = n * (n - 1) // 2
    if e > max_edges:
        raise ValueError(f"Too many edges: the maximum number of edges for {n} nodes is {max_edges}")

    # Start by creating a spanning tree to ensure the graph is connected
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(n - 1):
        G.add_edge(nodes[i], nodes[i + 1])

    # Calculate remaining edges to add
    remaining_edges = e - (n - 1)

    # Generate all possible edges excluding those already in the graph
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not G.has_edge(u, v)]

    # Randomly sample remaining edges from the possible edges
    if remaining_edges > 0:
        selected_edges = random.sample(all_possible_edges, remaining_edges)
        G.add_edges_from(selected_edges)

    return G


def cops_init_heuristic(graph: nx.Graph, number_of_cops, centrality_measure=nx.betweenness_centrality) -> list[int]:
    """Computes initial placement for the cops, based on the topology of the graph of the game. The best move for the cops is computed by means of a centrality heuristic, that denies the positions with the highest centrality. Thus it aims to separate the reachable graph for the robber into different components.

        :param graph: The underlying graph of the game.
        :param centrality_measure: The centrality measure that is used to evaluate a position. The default is betweeness centrality.
        :param number_of_cops: The number of cops in that need to be placed.
        :return: The list of the initial cop positions.
        """

    nodes_by_centrality = sorted(centrality_measure(graph).items(), key=lambda x: x[1], reverse=True)

    positions = [x[0] for x in nodes_by_centrality]

    for p in positions[1:]:
        if random.random() < float(0.3):
            if len(positions) > number_of_cops:
                break
            positions.remove(p)

    cop_positions = positions[:number_of_cops]

    return cop_positions


def generate_possible_moves_robber(graph: nx.Graph, old_cop_positions, cops_positions: tuple[int],
                                   robber_position: int) -> \
        set[int]:
    cops_staying = set(old_cop_positions) & set(cops_positions)
    copless_graph = graph.subgraph(set(graph) - (set(cops_staying) - set([robber_position])))
    robber_component = nx.node_connected_component(copless_graph, robber_position)
    return robber_component - set(cops_positions)


def robber_possible_moves(graph: nx.Graph, cop_positions: tuple[int], robber_position: int,
                          previous_cop_positions: tuple[int]) -> list[int]:
    """
    Return the possible moves for the robber, optimized for performance.
    Time complexity: O(n + m) where n is the number of nodes in the graph and m is the number of edges.
    """
    # Compute the controlled nodes once
    # Time complexity: O(k) where k is the number of cops (O(k) + O(k) + O(k))
    controlled_nodes = set(cop_positions) & set(previous_cop_positions)

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
    reachable_nodes = visited - set(cop_positions)

    return list(reachable_nodes)


if __name__ == '__main__':
    n = 96  # Horton
    n = 78  # Cactus 0078
    # number_cops = 9
    g = nx.Graph()
    g.add_nodes_from(range(n))

    ## Horton
    # edges = [{"source": 0, "target": 1}, {"source": 0, "target": 2}, {"source": 0, "target": 3}, {"source": 1, "target": 36}, {"source": 1, "target": 38}, {"source": 2, "target": 64}, {"source": 2, "target": 66}, {"source": 3, "target": 92}, {"source": 3, "target": 94}, {"source": 4, "target": 5}, {"source": 4, "target": 6}, {"source": 4, "target": 7}, {"source": 5, "target": 22}, {"source": 5, "target": 26}, {"source": 6, "target": 50}, {"source": 6, "target": 54}, {"source": 7, "target": 78}, {"source": 7, "target": 82}, {"source": 8, "target": 9}, {"source": 8, "target": 10}, {"source": 8, "target": 11}, {"source": 9, "target": 34}, {"source": 9, "target": 39}, {"source": 10, "target": 62}, {"source": 10, "target": 67}, {"source": 11, "target": 90}, {"source": 11, "target": 95}, {"source": 12, "target": 13}, {"source": 12, "target": 14}, {"source": 12, "target": 15}, {"source": 13, "target": 16}, {"source": 13, "target": 17}, {"source": 14, "target": 20}, {"source": 14, "target": 23}, {"source": 15, "target": 32}, {"source": 15, "target": 37}, {"source": 16, "target": 18}, {"source": 16, "target": 19}, {"source": 17, "target": 24}, {"source": 17, "target": 26}, {"source": 18, "target": 20}, {"source": 18, "target": 21}, {"source": 19, "target": 23}, {"source": 19, "target": 25}, {"source": 20, "target": 22}, {"source": 21, "target": 26}, {"source": 21, "target": 28}, {"source": 22, "target": 27}, {"source": 23, "target": 24}, {"source": 24, "target": 27}, {"source": 25, "target": 36}, {"source": 25, "target": 39}, {"source": 27, "target": 28}, {"source": 28, "target": 29}, {"source": 29, "target": 30}, {"source": 29, "target": 31}, {"source": 30, "target": 32}, {"source": 30, "target": 33}, {"source": 31, "target": 35}, {"source": 31, "target": 37}, {"source": 32, "target": 34}, {"source": 33, "target": 38}, {"source": 33, "target": 39}, {"source": 34, "target": 35}, {"source": 35, "target": 36}, {"source": 37, "target": 38}, {"source": 40, "target": 41}, {"source": 40, "target": 42}, {"source": 40, "target": 43}, {"source": 41, "target": 44}, {"source": 41, "target": 45}, {"source": 42, "target": 48}, {"source": 42, "target": 51}, {"source": 43, "target": 60}, {"source": 43, "target": 65}, {"source": 44, "target": 46}, {"source": 44, "target": 47}, {"source": 45, "target": 52}, {"source": 45, "target": 54}, {"source": 46, "target": 48}, {"source": 46, "target": 49}, {"source": 47, "target": 51}, {"source": 47, "target": 53}, {"source": 48, "target": 50}, {"source": 49, "target": 54}, {"source": 49, "target": 56}, {"source": 50, "target": 55}, {"source": 51, "target": 52}, {"source": 52, "target": 55}, {"source": 53, "target": 64}, {"source": 53, "target": 67}, {"source": 55, "target": 56}, {"source": 56, "target": 57}, {"source": 57, "target": 58}, {"source": 57, "target": 59}, {"source": 58, "target": 60}, {"source": 58, "target": 61}, {"source": 59, "target": 63}, {"source": 59, "target": 65}, {"source": 60, "target": 62}, {"source": 61, "target": 66}, {"source": 61, "target": 67}, {"source": 62, "target": 63}, {"source": 63, "target": 64}, {"source": 65, "target": 66}, {"source": 68, "target": 69}, {"source": 68, "target": 70}, {"source": 68, "target": 71}, {"source": 69, "target": 72}, {"source": 69, "target": 73}, {"source": 70, "target": 76}, {"source": 70, "target": 79}, {"source": 71, "target": 88}, {"source": 71, "target": 93}, {"source": 72, "target": 74}, {"source": 72, "target": 75}, {"source": 73, "target": 80}, {"source": 73, "target": 82}, {"source": 74, "target": 76}, {"source": 74, "target": 77}, {"source": 75, "target": 79}, {"source": 75, "target": 81}, {"source": 76, "target": 78}, {"source": 77, "target": 82}, {"source": 77, "target": 84}, {"source": 78, "target": 83}, {"source": 79, "target": 80}, {"source": 80, "target": 83}, {"source": 81, "target": 92}, {"source": 81, "target": 95}, {"source": 83, "target": 84}, {"source": 84, "target": 85}, {"source": 85, "target": 86}, {"source": 85, "target": 87}, {"source": 86, "target": 88}, {"source": 86, "target": 89}, {"source": 87, "target": 91}, {"source": 87, "target": 93}, {"source": 88, "target": 90}, {"source": 89, "target": 94}, {"source": 89, "target": 95}, {"source": 90, "target": 91}, {"source": 91, "target": 92}, {"source": 93, "target": 94}]

    ## Cactus 0078
    edges = [{"source": 0, "target": 1}, {"source": 0, "target": 3}, {"source": 0, "target": 72},
             {"source": 0, "target": 74}, {"source": 1, "target": 2}, {"source": 1, "target": 13},
             {"source": 1, "target": 15}, {"source": 2, "target": 3}, {"source": 2, "target": 4},
             {"source": 2, "target": 5}, {"source": 2, "target": 18}, {"source": 2, "target": 19},
             {"source": 3, "target": 6}, {"source": 3, "target": 8}, {"source": 4, "target": 5},
             {"source": 4, "target": 9}, {"source": 4, "target": 70}, {"source": 6, "target": 7},
             {"source": 6, "target": 60}, {"source": 7, "target": 8}, {"source": 7, "target": 16},
             {"source": 7, "target": 17}, {"source": 8, "target": 20}, {"source": 8, "target": 22},
             {"source": 9, "target": 10}, {"source": 10, "target": 11}, {"source": 10, "target": 12},
             {"source": 10, "target": 68}, {"source": 10, "target": 69}, {"source": 11, "target": 12},
             {"source": 11, "target": 65}, {"source": 11, "target": 67}, {"source": 13, "target": 14},
             {"source": 13, "target": 33}, {"source": 13, "target": 35}, {"source": 14, "target": 15},
             {"source": 15, "target": 43}, {"source": 15, "target": 45}, {"source": 15, "target": 50},
             {"source": 15, "target": 51}, {"source": 16, "target": 17}, {"source": 16, "target": 38},
             {"source": 16, "target": 39}, {"source": 17, "target": 26}, {"source": 17, "target": 28},
             {"source": 17, "target": 36}, {"source": 17, "target": 37}, {"source": 18, "target": 19},
             {"source": 18, "target": 29}, {"source": 18, "target": 30}, {"source": 19, "target": 31},
             {"source": 19, "target": 32}, {"source": 20, "target": 21}, {"source": 20, "target": 23},
             {"source": 20, "target": 25}, {"source": 20, "target": 58}, {"source": 20, "target": 71},
             {"source": 21, "target": 22}, {"source": 23, "target": 24}, {"source": 24, "target": 25},
             {"source": 25, "target": 46}, {"source": 25, "target": 47}, {"source": 26, "target": 27},
             {"source": 27, "target": 28}, {"source": 29, "target": 30}, {"source": 31, "target": 32},
             {"source": 31, "target": 59}, {"source": 32, "target": 61}, {"source": 32, "target": 63},
             {"source": 33, "target": 34}, {"source": 34, "target": 35}, {"source": 36, "target": 40},
             {"source": 36, "target": 42}, {"source": 38, "target": 39}, {"source": 40, "target": 41},
             {"source": 41, "target": 42}, {"source": 41, "target": 48}, {"source": 41, "target": 49},
             {"source": 41, "target": 64}, {"source": 43, "target": 44}, {"source": 43, "target": 55},
             {"source": 43, "target": 57}, {"source": 44, "target": 45}, {"source": 46, "target": 47},
             {"source": 48, "target": 49}, {"source": 50, "target": 51}, {"source": 50, "target": 52},
             {"source": 50, "target": 54}, {"source": 52, "target": 53}, {"source": 53, "target": 54},
             {"source": 55, "target": 56}, {"source": 56, "target": 57}, {"source": 61, "target": 62},
             {"source": 61, "target": 75}, {"source": 61, "target": 77}, {"source": 62, "target": 63},
             {"source": 65, "target": 66}, {"source": 66, "target": 67}, {"source": 68, "target": 69},
             {"source": 72, "target": 73}, {"source": 73, "target": 74}, {"source": 75, "target": 76},
             {"source": 76, "target": 77}]

    g.add_edges_from([(e["source"], e["target"]) for e in edges])
    timeout_init = 9.6
    timeout_step = 0.96
    max_rounds = 50

    TIMEOUTS_INIT_ROBBER: dict[float, float] = {2: 0.02, 3: 0.02, 4: 0.03, 5: 0.09, 6: 0.35, 7: 1.77, 8: 761.9}

    # cops = [54,55,5,56,57,6,73,26,12] # Horton
    cops = [0, 1, 3]  # Cactus 0078

    number_cops = len(cops)

    start_time = time.time()

    if 1 < g.number_of_nodes() < 8 and timeout_init < TIMEOUTS_INIT_ROBBER[
        g.number_of_nodes()]:
        best_score = math.inf
        best_move = None
        for node in g.nodes:
            if node not in cops:
                score = final_minimax.minimax_rob(g, 0, (), tuple(cops), node, max_rounds,
                                                  number_cops)[0]
                if score < best_score:
                    best_score = score
                    best_move = node
        robber_position = best_move
        # print(f'r initial: {self.robber_position}')
        print(robber_position)
    else:
        robber_position = heuristics.robber_init_heuristic(g, cops)
        print(robber_position)

    print('Time:', time.time() - start_time)
