import numpy as np

class AntColonyKnapsack:
    def __init__(self, items, max_weight, n_ants, n_iterations, decay=0.95, alpha=1, beta=2):
        self.items = items
        self.max_weight = max_weight
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        best_solution = None
        all_time_max_value = float('-inf')

        for _ in range(self.n_iterations):
            solutions = self.gen_all_solutions()
            self.spread_pheromone(solutions)

            self.decay_pheromone()

            self.ant_update_pheromone(solutions)

            global_max_value, global_best_solution = self.global_update_pheromone(solutions)

            if global_max_value > all_time_max_value:
                all_time_max_value = global_max_value
                best_solution = global_best_solution

            if _ % 10 == 0:
                print('Iteration: {}, Best Value: {}'.format(_, all_time_max_value))

        return best_solution, all_time_max_value

    def spread_pheromone(self, solutions):
        for solution in solutions:
            total_value = sum(item[1] for item in solution)
            for item in solution:
                pheromone_deposit = (item[1] / total_value) * 1000
                item[2] += pheromone_deposit

    def decay_pheromone(self):
        for item in self.items:
            item[2] *= self.decay

    def ant_update_pheromone(self, solutions):
        for solution in solutions:
            total_value = sum(item[1] for item in solution)
            for item in solution:
                item[2] += (item[1] / total_value) * 100

#The pheromone levels are updated based on the value of the selected items
    def global_update_pheromone(self, solutions):
        global_max_value = float('-inf')
        global_best_solution = None

        for solution in solutions:
            total_value = sum(item[1] for item in solution)
            if total_value > global_max_value:
                global_max_value = total_value
                global_best_solution = solution

        return global_max_value, global_best_solution

    def gen_all_solutions(self):
        all_solutions = []
        for _ in range(self.n_ants):
            solution = self.gen_solution()
            all_solutions.append(solution)
        return all_solutions

    def gen_solution(self):
        solution = []
        remaining_weight = self.max_weight

        while remaining_weight > 0:
            probabilities = self.calculate_probabilities(remaining_weight)
            selected_item_index = np_choice(range(len(self.items)), 1, p=probabilities)[0]
            selected_item = self.items[selected_item_index]

            if selected_item[0] <= remaining_weight:
                solution.append(selected_item)
                remaining_weight -= selected_item[0]

        return solution

    def calculate_probabilities(self, remaining_weight):
        probabilities = [item[2] ** self.alpha * ((item[1] / item[0]) ** self.beta) if item[0] <= remaining_weight else 0 for item in self.items]
        probabilities = np.array(probabilities) / sum(probabilities)
        return probabilities

def np_choice(a, size, replace=True, p=None):
    return np.random.choice(a, size=size, replace=replace, p=p)

if __name__ == "__main__":
    # Example usage
    np.random.seed(123)
    items = [
        [2, 5, 1],  # [weight, value, pheromone]
        [3, 8, 1],
        [5, 13, 1],
        [7, 20, 1],
        [1, 3, 1]
    ]
    max_weight = 8
    n_ants = 5
    n_iterations = 50

    ant_colony_knapsack = AntColonyKnapsack(items, max_weight, n_ants, n_iterations)
    best_solution, best_value = ant_colony_knapsack.run()

    print("Best Solution:", best_solution)
    print("Best Value:", best_value)
