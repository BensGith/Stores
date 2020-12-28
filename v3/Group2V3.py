import numpy as np


class Store:

    def __init__(self, number, cost_tbl, p, n):
        self.number = number
        self.index = number - 1
        self.score = 0
        self.n_of_stores = n
        self.next_store = (self.number % n) + 1
        if self.number == 1:
            self.prv_store = n
        else:
            self.prv_store = self.number - 1

        self.costs = [cost_tbl[self.index][i] for i in range(p)]
        self.p_cluster_size = p

    def __repr__(self):
        return "Store {} - {}".format(self.number, self.score)

    def cost_n_stores(self, steps):
        """
        get the cost of collecting this store + (steps -1) stores together
        for example if steps = 1, get the cost of collecting just this store to a cluster
        grab [0] element from list
        :param steps:
        :return:
        """
        return self.costs[steps - 1]

    def n_forward(self, n):
        """
        calculates the n stores in front
        :param n: number of steps MUST BE SMALLER THAN OR EQUAL N number of stores
        :return: list of store numbers
        """
        return [self.number + i if self.number + i <= self.n_of_stores
                else (self.number + i) % self.n_of_stores for i in range(1, n + 1)]


class TSP:
    def __init__(self, start_store, number_of_shops_N, number_of_cars_L, max_cluster_size_P, cost_table):
        self.start_store = start_store
        self.s = self.start_store - 1
        self.n = number_of_shops_N
        self.p = max_cluster_size_P
        self.l = number_of_cars_L
        self.cost_table = cost_table
        self.stores = [Store(i, cost_table, max_cluster_size_P, self.n) for i in range(1, self.n + 1)]
        self.mat = np.ones((self.n, self.n + 1)) * float('inf')  # we have n stores and n+1 possible stores left
        self.mat[self.s][0] = 0  # s(n+1,0) = 0
        self.optimal_solution = float('inf')
        self.best_path = {}

    def __lt__(self, other):
        return self.optimal_solution < other.optimal_solution

    def solve(self):
        store_order = list(map(lambda z: z-1, self.stores[self.s].n_forward(self.n-1)))[::-1]+[self.s]
        for i in store_order:
            for j in range(n+1):  # left stores
                if j == 0 and i == self.s:
                    continue
                options = [float('inf')]
                for x in range(1, p+1):
                    if j - x < 0:  # minus stores left isn't possible
                        continue
                    possible_cluster = self.stores[i].n_forward(x)
                    origin = possible_cluster[-1]-1
                    cost = self.cost_of_cluster(i, x)
                    options.append(cost + self.mat[origin][j-x])
                self.mat[i][j] = min(options)
        self.optimal_solution = self.mat[self.s][-1]

    def restore_solution(self):
        print("Optimal Solution is {}".format(self.optimal_solution))

    def cost_of_cluster(self, store_n, x):
        return self.stores[store_n].costs[x - 1]


def open_file(path):
    with open(path, 'r') as file:
        given_data = file.readlines()
    cost_table = []
    number_of_shops_N = None
    number_of_cars_L = None
    max_cluster_size_P = None
    for i in range(len(given_data)):
        if i == 0:
            number_of_shops_N = int(given_data[i])
        elif i == 1:
            number_of_cars_L = int(given_data[i])
        elif i == 2:
            max_cluster_size_P = int(given_data[i])
        else:
            cost_table.append(list(map(int, given_data[i].strip().split(","))))
    return number_of_shops_N, number_of_cars_L, max_cluster_size_P, cost_table


if __name__ == "__main__":
    n, l, p, costs = open_file("large.txt")
    optional_solutions = []
    for i in range(1, n+1):
        tsp1 = TSP(i, n, l, p, costs)
        tsp1.solve()
        optional_solutions.append(tsp1)
    optimal_solution = min(optional_solutions)
    optimal_solution.restore_solution()

