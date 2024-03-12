import random
import matplotlib.pyplot as plt

def generate_random_solution():
    # Generate a random individual for the initial population. 
    return [random.randint(0, 7) for _ in range(8)]

def calculate_fitness(solution):
    # Calculate the fitness of a solution based on the number of non-attacking pairs of queens.
    non_attacking_pairs = 28
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if solution[i] == solution[j] or abs(solution[i] - solution[j]) == j - i:
                non_attacking_pairs -= 1
    return non_attacking_pairs

def initialize_population(population_size):
    # Initialize the population with a given number of random solutions.
    return [(generate_random_solution(), 0) for _ in range(population_size)]

def compute_population_fitness(population):
    # Compute and store the fitness for each solution in the population
    return [(solution, calculate_fitness(solution)) for solution, _ in population]

def select_parents(population):
    # Select parent solutions from the population based on normalized fitness probabilities.
    total_fitness = sum(fitness for _, fitness in population)
    selection_probabilities = [fitness / total_fitness for _, fitness in population]
    parents = random.choices(
        population, 
        weights=selection_probabilities, 
        k=2
    )
    return parents

def crossover(parents):
    # Perform a single-point crossover between two parents to produce two children.
    crossover_point = random.randint(1, 7)
    parent1, parent2 = parents
    child1 = parent1[0][:crossover_point] + parent2[0][crossover_point:]
    child2 = parent2[0][:crossover_point] + parent1[0][crossover_point:]
    return child1, child2

def mutate(child, mutation_pct):
    if random.random() < mutation_pct:
        mutation_point = random.randint(0, 7)
        original_value = child[mutation_point]
        child[mutation_point] = random.choice([i for i in range(8) if i != original_value])
    return child

def genetic_algorithm(population_size, num_iterations, mutation_pct, fitness_threshold):
    # Initialize the population with random solutions and calculate their fitness
    population = initialize_population(population_size)
    population = compute_population_fitness(population)
    
    avg_fitness_over_time = []
    best_fitness_over_time = []
    
    # Print the initial population and their fitness
    print("Initial Population and Fitness:")
    for solution, fitness in population:
        print(solution, fitness)
    
    best_solution = None
    best_iteration = 0
    
    for iteration in range(num_iterations):
        # Compute average fitness and check the stopping condition
        average_fitness = sum(fitness for _, fitness in population) / population_size
        avg_fitness_over_time.append(average_fitness)
        
        current_best_solution = max(population, key=lambda x: x[1])
        best_fitness_over_time.append(current_best_solution[1])
        if not best_solution or current_best_solution[1] > best_solution[1]:
            best_solution = current_best_solution
            best_iteration = iteration

        if average_fitness > fitness_threshold:
            print(f"Stopping early at iteration {iteration} with average fitness: {average_fitness}")
            break

        # Selection, crossover, mutation, and fitness calculation for new offspring
        new_population = []
        for _ in range(population_size // 2):
            # Selection of parents
            parents = select_parents(population)

            # Crossover
            child1, child2 = crossover(parents)

            # Mutation
            child1 = mutate(child1, mutation_pct)
            child2 = mutate(child2, mutation_pct)

            # Compute fitness for the new children
            new_population.append((child1, calculate_fitness(child1)))
            new_population.append((child2, calculate_fitness(child2)))

        # Replace the old population with the new population
        population = new_population
    
        # Update the best solution if a new best is found
        current_best_solution = max(population, key=lambda x: x[1])
        if not best_solution or current_best_solution[1] > best_solution[1]:
            best_solution = current_best_solution
            best_iteration = iteration
            
            
    # Plot the performance over time
    plt.figure(figsize=(12, 6))
    plt.plot(avg_fitness_over_time, label='Average Fitness', color='blue')
    plt.plot(best_fitness_over_time, label='Best Fitness', color='green')
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness')
    plt.title('Genetic Algorithm Performance')
    plt.legend()
    plt.show()
    
    # Return the best solution from the final population along with the iteration it was found
    return best_solution, best_iteration


# Parameters
PopulationSize = 100
NumIterations = 1000
MutationPct = 0.05
FitnessThreshold = 27


best_solution, best_iteration = genetic_algorithm(PopulationSize, NumIterations, MutationPct, FitnessThreshold)
print(f"\nBest solution found at iteration {best_iteration}:")
print(best_solution[0])
print("Fitness:", best_solution[1])
