import functools
import random

from networkx import Graph, degree_centrality, shortest_path, number_of_nodes, betweenness_centrality, \
    eigenvector_centrality_numpy, connected_components, eigenvector_centrality, shortest_path_length

import final_minimax as minimax

# choose optimization: global variables
ROBBER_CHECK_FOR_MOVING_COPS_ACTIVE = False


# different helper functions

# copied from bot_easy:
# graph without the positions blocked by cops
def get_copless_graph(graph: Graph, cop_positions: list[int], robber_position: int) -> Graph:
    return graph.subgraph(set(graph) - (set(cop_positions) - set([robber_position])))


# copied from bot_easy:
# check if lost
def is_lost(cop_positions: list[int], robber_position: int) -> bool:
    if robber_position in cop_positions:
        return True
    else:
        return False


# the subgraph that is reachable from the robbers position
@functools.cache
def reachable_nodes(graph: Graph, cop_positions: tuple[int], robber_position):
    return set(graph.subgraph(
        shortest_path(get_copless_graph(graph, cop_positions, robber_position), robber_position)).nodes) - set(
        cop_positions)


# the subgraph that is reachable from the robbers position
def reachable_nodes_plus_cops(graph, cop_positions, robber_position):
    return set(graph.subgraph(
        shortest_path(get_copless_graph(graph, cop_positions, robber_position), robber_position)).nodes)


# heuristics, here only for robber but normalized to 0 to 1

# the more complex centrality based heuristics better not on sigluar nodes but later in the application
# global evaluation since for example eigen centrality is always computed for a whole graph

# simple heuristic: portion of reachable nodes -> between 0 and 1
def reachable_heuristic(graph, cop_positions, robber_position) -> int:
    if is_lost(cop_positions, robber_position):
        return 0

    return reachable_nodes(graph, tuple(cop_positions), robber_position) / number_of_nodes(graph)


# simple heuristic: degree centrality of the robber pos
# also for other networkx centralities
def degree_centrality_heuristic(graph: Graph, cop_positions, robber_position) -> int:
    if is_lost(cop_positions, robber_position):
        return 0

    return degree_centrality(get_copless_graph(graph, cop_positions, robber_position))[robber_position]


# game initialization
# use heuristics to determine the initial positions

# initialize game for cops
def cops_init_heuristic(graph: Graph, number_of_cops, centrality_measure=betweenness_centrality) -> list[int]:
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


# initialize game for robber
def robber_init_heuristic(graph: Graph, cop_positions: list[int], centrality_measure=eigenvector_centrality) -> int:
    """Computes initial position for the robber, based on the initial placement of the cops. The best step is computed by means of a centrality heuristic combined with an evaluation of the reachable nodes of each component of the graph, if the cops were able to seperate the graph with their initial placement. The heuristic evaluates a position according to a centrality measure.

        :param graph: The underlying graph of the game
        :param cop_positions: The initial cop positions.
        :param centrality_measure: The centrality measure that is used to evaluate the positions in the graph. The default is eigenvector centrality that only works for connected graphs.
        :return: The initial robber position.
        """
    copless_graph = graph.subgraph(set(graph) - (set(cop_positions)))

    global_pos = dict.fromkeys(graph, -100)

    components = connected_components(copless_graph)

    for c in components:
        if centrality_measure == eigenvector_centrality:
            centrality = centrality_measure(graph.subgraph(c), max_iter=1000)
        else:
            centrality = centrality_measure(graph.subgraph(c))
        size = len(c) / graph.number_of_nodes()
        for i in c:
            global_pos[i] = centrality[i] * size

    best_pos = sorted(global_pos.keys(), key=global_pos.get, reverse=True)[0]
    return best_pos


# initilize game for the robber with a simpler, less computationally complex heuristic
def robber_init_heuristic_simple(graph: Graph, cop_positions, centrality_measure=eigenvector_centrality_numpy) -> int:
    """Computes initial position for the robber, based on the initial placement of the cops. The best step is computed by means of a simple centrality heuristic, that chooses the position with the highest centrality.

        :param graph: The underlying graph of the game
        :param cop_positions: The initial cop positions.
        :param centrality_measure: The centrality measure that is used to evaluate the positions in the graph. The default is eigenvector centrality that only works for connected graphs.
        :return: The initial robber position.
        """
    copless_graph = graph.subgraph(set(graph) - set(cop_positions))

    best_pos = max(eigenvector_centrality_numpy(copless_graph), key=lambda x: x[1])[0]

    return best_pos


# game step
# use heuristics to determine next move

# decide the next move for the robber
def robber_heuristic_step(graph: Graph, current_rob_position, old_cops_positions: list[int],
                          cops_positions: list[int]) -> int:
    """Computes the next step for the robber, based on the current position of the cops and the robber position. The best step is computed by means of a centrality heuristic, that chooses the position with the highest centrality that is reachable for the robber.

        :param graph: The underlying graph of the game
        :param old_cops_positions: The initial cop positions.
        :param cops_positions: The positions the cops are moving to.
        :param current_rob_position: The current position of the robber
        :return: The next robber position.
        """
    # get reachable part of the graph
    reachable_nodes = minimax.generate_possible_moves_robber(graph, tuple(old_cops_positions), tuple(cops_positions),
                                                             current_rob_position)

    # search best reachable pos
    if len(reachable_nodes) > 1:
        evaluate_pos = sorted(betweenness_centrality(graph.subgraph(reachable_nodes)), reverse=True)
    elif len(reachable_nodes) == 1:
        return reachable_nodes.pop()
    else:
        return current_rob_position

    if len(evaluate_pos) > 1:
        if graph.degree(evaluate_pos[0]) >= graph.degree(evaluate_pos[1]):
            return evaluate_pos[0]
        else:
            return evaluate_pos[1]


# decide the next move for the cops
def cops_heuristic_step(graph: Graph, current_cops_pos: list[int], rob_pos, max_rounds) -> list[int]:
    """Computes the next step for the cops, based on the current position of the cops and the robber position. The best move for the cops is computed by means of a centrality heuristic, that denies the positions with the highest centrality that is reachable for the robber.

        :param graph: The underlying graph of the game
        :param current_cops_pos: The initial cop positions.
        :param rob_pos: The  position of the robber
        :param max_rounds: The max number of rounds the game is played for. If the cops cannot catch the robber in this time, they lose.
        :return: The list of the next cop positions.
        """

    # check if there are cops that are not useful
    unused_cops = []

    for cop in current_cops_pos:
        for pos in shortest_path(graph, rob_pos, cop):
            if pos in current_cops_pos and pos != cop:
                unused_cops.append(cop)
        if graph.degree(cop) < 2 and cop not in unused_cops:
            unused_cops.append(cop)

    # evaluate the positions that the robber can go to
    robber_reachable = reachable_nodes(graph, tuple(current_cops_pos), rob_pos)
    evaluate_pos = sorted(betweenness_centrality(graph.subgraph(robber_reachable - set(current_cops_pos))).items(),
                          key=lambda x: x[1], reverse=True)

    # move unused cops
    new_cop_positions = []
    if len(unused_cops) > 0:
        for cop in unused_cops:
            if len(evaluate_pos) == 0:
                new_cop_positions.append(cop)
            else:
                pos = evaluate_pos.pop(0)[0]
                new_cop_positions.append(pos)

        cop_positions = [cop if cop not in unused_cops else new_cop_positions.pop(0) for cop in current_cops_pos]
        return cop_positions

    # if there are no unused cops

    # if there are only few remaining accesable nodes for the robber use bruteforce
    elif False and len(robber_reachable) < 3:
        node_set = reachable_nodes(graph, current_cops_pos, rob_pos).union(current_cops_pos).union(
            [n in graph.neighbors(n) for n in current_cops_pos])
        reduced_graph = graph.subgraph(node_set)
        res = minimax.find_best_move_cops(reduced_graph, max_rounds, len(current_cops_pos), tuple(current_cops_pos),
                                          rob_pos)
        return res

    # if the robber still has a large number of accessable nodes
    else:
        # sort cops according to dist to the robber
        distances = dict(
            [(cop_pos, shortest_path_length(graph, source=rob_pos, target=cop_pos)) for cop_pos in current_cops_pos])
        sorted_cops = sorted(distances.items(), key=lambda x: x[1], reverse=True)

        # sort cops according to betweeness in the whole graph
        # sorted_cops = betweenness_centrality(graph)
        # sorted_cops = {key: sorted_cops[key] for key in current_cops_pos}
        # sorted_cops = sorted(sorted_cops.items(), key=lambda x: x[1], reverse=False)

        # move a third of the number of cops, beginning with those that are the furthest away
        new_cop_positions = []
        moved = 0

        for (cop, dist) in sorted_cops:
            if (moved > len(current_cops_pos) / 3) & (not moved == 0):
                new_cop_positions.append(cop)
            else:
                if len(evaluate_pos) == 0:
                    new_cop_positions.append(cop)
                else:
                    pos = evaluate_pos.pop(0)[0]
                    new_cop_positions.append(pos)
                    moved += 1

        cop_positions = new_cop_positions
        return cop_positions
