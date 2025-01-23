import itertools
import random

import matplotlib.pyplot as plt
import networkx as nx
from numpy.ma.extras import average

import minimax_iterative
import minimax_iterative_ex
import old_minimax
import old_minimax_changed
import time
import new_minimax

import dump_00002
import dump_00001
import dump_00003
import dump_00004
import dump_00005

import final_minimax

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

def generate_possible_moves_robber(graph: nx.Graph, old_cop_positions, cops_positions: tuple[int], robber_position: int) -> \
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
    # n = 10
    # e = 15
    # g = nx.Graph()
    # g.add_nodes_from(range(n))
    # g.add_edges_from(
    #     [(0, 3), (0, 5), (0, 6), (0, 7), (1, 5), (1, 8), (2, 3), (2, 4), (2, 6), (2, 7), (3, 8), (3, 9), (4, 6), (6, 8),
    #      (7, 8)])
    # for _ in range(30):

    # g = create_random_connected_graph(8, 13)
    # nx.draw(g, with_labels=True)
    # plt.show()

    #

    import logging
    import math

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename= 'main.log', encoding='utf-8', level=logging.INFO, filemode='w')

    summary = {}

    for n in range(2,11):
        edges = n * (n - 1) // 2
        cops_count = n//2
        max_rounds = 20
        times = []
        for e in range(edges):
            g = create_random_connected_graph(n, e)
            start_time = time.perf_counter()

            possible_positions = itertools.combinations(g.nodes, cops_count)
            possible_robbers = set(g.nodes)
            best_score = -math.inf
            best_move = None
            best_move_r = 0
            alpha = -math.inf
            beta = math.inf
            for position in possible_positions:
                for robber in possible_robbers:
                    if not robber in position:
                        best_move_r = final_minimax.find_best_move_robber(g, max_rounds, cops_count, (),
                                                                    position, robber, 0, 0)
                score = final_minimax.minimax_cop(g, 0, position, best_move_r, max_rounds, cops_count)[0]
                if score > best_score:
                    best_score = score
                    best_move = position

            cop_positions = list(best_move)
            end_time = time.perf_counter()
            times.append(end_time-start_time)
            logging.info(f"Nodes: {n}, Edges: {e}, Time: {end_time-start_time}")
        summary[n] = f'{min(times): .2f}, {average(times): .2f}, {max(times): .2f}'
    print(summary)
