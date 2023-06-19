import random
import networkx as nx
import matplotlib.pyplot as plt


graph = [
    [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    [10, 0, 35, 25, 30, 45, 50, 55, 40, 60, 20, 15, 65, 70, 75, 80, 85, 90, 95, 100],
    [15, 35, 0, 30, 45, 40, 60, 20, 55, 10, 50, 25, 65, 70, 75, 80, 85, 90, 95, 100],
    [20, 25, 30, 0, 60, 50, 55, 35, 10, 45, 15, 40, 65, 70, 75, 80, 85, 90, 95, 100],
    [25, 30, 45, 60, 0, 20, 25, 30, 35, 40, 55, 50, 65, 70, 75, 80, 85, 90, 95, 100],
    [30, 45, 40, 50, 20, 0, 35, 40, 50, 55, 10, 15, 65, 70, 75, 80, 85, 90, 95, 100],
    [35, 50, 60, 55, 25, 35, 0, 15, 30, 45, 20, 10, 65, 70, 75, 80, 85, 90, 95, 100],
    [40, 55, 20, 35, 30, 40, 15, 0, 50, 10, 25, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    [45, 40, 55, 10, 35, 50, 30, 50, 0, 15, 60, 25, 65, 70, 75, 80, 85, 90, 95, 100],
    [50, 60, 10, 45, 40, 55, 45, 10, 15, 0, 35, 20, 65, 70, 75, 80, 85, 90, 95, 100],
    [55, 20, 50, 15, 55, 10, 20, 25, 60, 35, 0, 45, 65, 70, 75, 80, 85, 90, 95, 100],
    [60, 15, 25, 40, 50, 15, 10, 60, 25, 20, 45, 0, 65, 70, 75, 80, 85, 90, 95, 100],
    [65, 70, 75, 80, 85, 90, 95, 100, 65, 70, 75, 80, 0, 10, 15, 20, 25, 30, 35, 40],
    [70, 75, 80, 85, 90, 95, 100, 65, 70, 75, 80, 85, 10, 0, 35, 25, 30, 45, 50, 55],
    [75, 80, 85, 90, 95, 100, 65, 70, 75, 80, 85, 90, 15, 35, 0, 30, 45, 40, 60, 20],
    [80, 85, 90, 95, 100, 65, 70, 75, 80, 85, 90, 95, 20, 25, 30, 0, 60, 50, 55, 35],
    [85, 90, 95, 100, 65, 70, 75, 80, 85, 90, 95, 100, 25, 30, 45, 60, 0, 20, 25, 30],
    [90, 95, 100, 65, 70, 75, 80, 85, 90, 95, 100, 65, 30, 45, 40, 50, 20, 0, 35, 40],
    [95, 100, 65, 70, 75, 80, 85, 90, 95, 100, 65, 70, 35, 50, 60, 55, 25, 35, 0, 15],
    [100, 65, 70, 75, 80, 85, 90, 95, 100, 65, 70, 75, 40, 55, 20, 35, 30, 40, 15, 0]]


pop_size = 1000
mutation_rate = 0.1
num_generations = 500


def create_population(pop_size, graph):
    population = []
    num_nodes = len(graph)
    for _ in range(pop_size):
        path = list(range(num_nodes))
        random.shuffle(path)
        population.append(path)
    return population


def calculate_fitness(path, graph):
    fitness = 0
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i+1]
        edge_weight = graph[start_node][end_node]  
        fitness += edge_weight
    return fitness



def select_parents(population, tournament_size):
    parents = []
    for _ in range(2):
        tournament = random.sample(population, tournament_size)
        winner = min(tournament, key=lambda x: calculate_fitness(x, graph))
        parents.append(winner)
    return parents

def crossover(parents):
    child = [-1] * len(parents[0])
    start, end = sorted(random.sample(range(len(parents[0])), 2))
    child[start:end+1] = parents[0][start:end+1]
    for i in range(len(parents[1])):
        if parents[1][i] not in child:
            for j in range(len(child)):
                if child[j] == -1:
                    child[j] = parents[1][i]
                    break
    return child


def mutate(path, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(path)), 2)
        path[idx1], path[idx2] = path[idx2], path[idx1]
    return path

def create_next_generation(population, tournament_size, mutation_rate):
    next_generation = []
    while len(next_generation) < len(population):
        parents = select_parents(population, tournament_size)
        child = crossover(parents)
        child = mutate(child, mutation_rate)
        next_generation.append(child)
    return next_generation


def find_shortest_path(graph, pop_size, mutation_rate, num_generations):
    population = create_population(pop_size, graph)
    for _ in range(num_generations):
        population = create_next_generation(population, 5, mutation_rate)
    best_path = min(population, key=lambda x: calculate_fitness(x, graph))
    return best_path


def draw_graph_with_best_path(graph, best_path):
    pos = nx.shell_layout(nx_graph)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12)) 

    
    node_colors = ['lightblue' if node != best_path[0] and node != best_path[-1] else 'grey' for node in nx_graph.nodes]
    nx.draw(nx_graph, pos, with_labels=False, node_color=node_colors, node_size=1000, font_size=12, ax=ax1)

  
    start_label = "Start"
    end_label = "End"
    node_labels = {best_path[0]: start_label, best_path[-1]: end_label}
    nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_color='black', font_size=12, font_weight='bold',
                            verticalalignment='center', horizontalalignment='center', ax=ax1)

    other_labels = {node: str(node) for node in nx_graph.nodes if node != best_path[0] and node != best_path[-1]}
    nx.draw_networkx_labels(nx_graph, pos, labels=other_labels, font_color='black', font_size=10,
                            verticalalignment='center', horizontalalignment='center', ax=ax1)

    
    best_path_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
    best_path_edges.append((best_path[-1], best_path[0]))  
    nx.draw_networkx_edges(nx_graph, pos, edgelist=best_path_edges, edge_color='red', width=2.0, ax=ax1)

   
    edge_labels = nx.get_edge_attributes(nx_graph, 'weight')
    best_path_edge_labels = {(u, v): edge_labels[(u, v)] if (u, v) in edge_labels else edge_labels[(v, u)]
                             for (u, v) in best_path_edges}
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=best_path_edge_labels, font_color='black',
                                 font_size=8, font_weight='bold', label_pos=0.5, rotate=False, ax=ax1)

    ax1.set_title('Graph with Best Path')
    ax1.axis('off')  

    
    output_text = f"Best path: {best_path}\nFitness: {calculate_fitness(best_path, graph)}"
    ax2.text(0.5, 0.5, output_text, fontsize=12, ha='center')
    ax2.axis('off')  

    plt.tight_layout()  
    plt.show()



best_path = find_shortest_path(graph, pop_size, mutation_rate, num_generations)
print("Best path:", best_path)
print("Fitness:", calculate_fitness(best_path, graph))


nx_graph = nx.Graph()
for i in range(len(graph)):
    for j in range(i + 1, len(graph)):
        if graph[i][j] != 0:
            nx_graph.add_edge(i, j, weight=graph[i][j])

draw_graph_with_best_path(graph, best_path)
