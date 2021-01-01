import math


class Store:

    def __init__(self, number, cost_tbl, p, n):
        self.number = number
        self.index = number - 1  # fix
        self.n_of_stores = n
        self.costs = [cost_tbl[self.index][i] for i in range(p)]

    def __repr__(self):
        return "Store {} ".format(self.number)

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


class State:
    """
    class to store the best option for each Belman equation
    this is helpful so we can call min() on all options and extract the cost and state in O(n)
    """
    def __init__(self, cost, state):
        self.cost = cost
        self.state = state

    def __lt__(self, other):
        return self.cost < other.cost

    def __repr__(self):
        return " Cost {} Origin State {}".format(self.cost, self.state)


class TSP:
    def __init__(self, start_store, number_of_shops_N, number_of_cars_L, max_cluster_size_P, cost_table):
        self.start_store = start_store
        self.s = self.start_store - 1
        self.n = number_of_shops_N
        self.p = max_cluster_size_P
        self.l = number_of_cars_L
        self.cost_table = cost_table
        self.min_clusters = math.ceil(self.n / self.p)
        self.stores = [Store(i, cost_table, max_cluster_size_P, self.n) for i in range(1, self.n + 1)]  # store objects
        # order of stores we will use for iterating from end to start
        self.compute_order = self.stores[self.s].n_forward(self.n - 1)[::-1] + [self.start_store]
        # order of stores starting from s
        self.store_order = [self.start_store] + self.stores[self.s].n_forward(self.n - 1)
        self.store_index = {self.store_order[j]: j for j in range(self.n)}  # index of stores starting from s
        self.best_path = {}  # store the path to solution as current state: next state
        self.solution = {}  # store value for each state as (i,j):cost
        self.optimal_solution = float('inf')
        self.optimal_start = None

    def __lt__(self, other):
        return self.optimal_solution < other.optimal_solution

    def store_ix(self, store, x):
        """
        Mi,x the store we get in X steps forward
        :param store:
        :param x:
        :return:
        """
        if store + x > self.n:
            return (store + x) % self.n
        return store + x

    def distance_ij(self, store1, store2):
        """
        gives the number of steps between store1 and store 2
        :param store1: origin store number
        :param store2: destination store number
        :return: x - number of steps from store1 to store2
        """

        if self.store_index[store1] > self.store_index[store2]:
            return (self.n - self.store_index[store1]) + self.store_index[store2]
        return self.store_index[store2] - self.store_index[store1]

    def init_solution(self):
        """
        setup a dictionary before solving the problem
        every possible ending state will be set to value of 0
        every non-possible ending state and all other states will be infinity
        :return:
        """
        for store in self.store_order:
            for j in range(self.l + 1):  # allowed states for left clusters 0 to number of cars
                if store == self.start_store:
                    if j <= self.l - self.min_clusters:  # allow remaining of spare clusters
                        self.solution[str((self.start_store, j))] = 0  # cost of possible endings is 0
                    else:
                        self.solution[str((self.start_store, j))] = float("inf")  # cost of impossible endings is inf
                else:  # set cost of other states to inf for now
                    self.solution[str((store, j))] = float("inf")

    def solve(self):
        """
        iterate over every store O(N)
        for every store, iterate over possible left clusters O(L)
        for every combinations of (i,j) - create P previous states and decide which one is the best. O(P)
        computing a decision is O(1)
        total complexity - O(NLP)
        :return:
        """
        self.init_solution()  # initiate edge cases
        for store in self.compute_order:
            for j in range(1, self.l + 1):  # number of clusters left
                # create P previous states (even non valid ones - they will get the value infinity!)
                possible_states = [State(self.cost_of_cluster(store, x) +
                                         self.solution[str((self.store_ix(store, x), j - 1))],
                                         str((self.store_ix(store, x), j - 1))) for x in range(1, self.p + 1)]
                best_state = min(possible_states)  # decide between P possibilities
                self.solution[str((store, j))] = best_state.cost  # store the cost of state
                # store in the path "parent" state leads to decided best
                self.best_path[str((store, j))] = best_state.state
                # we got to the last (first) store, save solution
                if store == self.start_store and self.optimal_solution > best_state.cost:
                    # save this to know the needed number of iterations to restore solution
                    # this is a PSEUDO starting point. we need this state to restore the best path
                    self.optimal_start = str((store, j))
                    self.optimal_solution = best_state.cost

    def restore_solution(self):
        """
        restore and print the optimal solution
        :return:
        """
        path = [self.optimal_start]  # add pseudo starting point to path
        current_state = self.optimal_start
        for i in range(convert_key(self.optimal_start)[1]):
            next_state = self.best_path[current_state]
            path.append(next_state)
            current_state = next_state
        clusters = []  # list of clusters
        for i in range(len(path) - 1):
            converted_state1 = convert_key(path[i])  # current state
            converted_state2 = convert_key(path[i + 1])  # next state
            # number of steps between the store that starts the cluster to the store that ends the cluster
            x = self.distance_ij(converted_state1[0], converted_state2[0]) - 1
            # add to cluster all stores in the range
            clusters.append([converted_state1[0]] + self.stores[converted_state1[0] - 1].n_forward(x))

        print("\nStarting From Store {}".format(self.start_store))
        print("\nOptimal Solution {}".format(self.optimal_solution))
        print("\nClusters")
        for c in clusters:
            print(c)

    def cost_of_cluster(self, store, x):
        """
        cost of cluster starting from store i moving x stores forward
        :param store: store
        :param x: number of steps
        :return: cost of cluster
        """
        return self.stores[store - 1].costs[x - 1]


def convert_key(key):
    """
    convert str(i,j) to tuple(int(i),int(j))
    :param key:
    :return:
    """
    return tuple(map(int, key[1:-1].split(",")))


def open_file(path):
    """
    open the input file in the specified path
    :param path: path of input file
    :return: n,l,p, cost table
    """
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

