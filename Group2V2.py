import numpy as np


class Store:

    def __init__(self, number, cost_tbl, p, n):
        self.number = number
        self.index = number - 1  # fix
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

    def __lt__(self, other):
        return self.score < other.score

    def __add__(self, other):
        return self.score + other.score

    def __radd__(self, other):
        return self.score + other

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

    def n_back(self, n):
        """
                calculates the n stores in front
                :param n: number of steps MUST BE SAMLLER OR EQUAL THAN N number of stores
                :return: list of store numbers
                """
        return [self.number - i if self.number - i >= 1
                else (self.number - i) + self.n_of_stores for i in range(1, n + 1)]


class TSP:
    def __init__(self, start_store, number_of_shops_N, number_of_cars_L, max_cluster_size_P, cost_table, direction):
        self.start_store = start_store
        self.s = self.start_store - 1
        self.n = number_of_shops_N
        self.p = max_cluster_size_P
        self.l = number_of_cars_L
        self.cost_table = cost_table
        self.direction = direction
        self.store_set = self.power_set()
        self.stores = [Store(i, cost_table, max_cluster_size_P, self.n) for i in range(1, self.n + 1)]
        self.position_indexes = {str(self.convert_to_bits(self.store_set[i])): i for i in range(len(self.store_set))}
        self.mat = np.ones((number_of_shops_N, 2 ** number_of_shops_N)) * float('inf')
        self.mat[self.s][0] = 0
        self.start_state = (self.start_store, str(self.convert_to_bits(self.store_set[0])), 0)
        self.best_path = {}
        self.optimal_solution = float('inf')


    def __lt__(self, other):
        return self.optimal_solution < other.optimal_solution

    def solve(self):
        options = self.prv_positions(self.start_state)
        final_positions = [(position[0],
                            position[1],
                            self.stores[position[0] - 1].costs[abs(self.n - calc_sum(position[1])) - 1]) for
                           position in options]
        for position in final_positions:  # insert final step values to matrix (1 step to s,[]) [1,1,1] is all visited
            self.mat[position[0]-1][self.position_indexes[position[1]]] = self.cost_of_cluster(self.start_state,
                                                                                               position)
        self.best_path = {position[1]: self.start_state[1] for position in final_positions}
        for i in range(2, self.l+1):  # main loop
            tmp_lst = []
            for position in final_positions:
                prv_pos_lst = (self.prv_positions(position))  # generate 2P possible positions,remove non valid ones
                tmp_lst += prv_pos_lst
                # for every previous position, decide if the path from it is better or not
                for prv_pos in prv_pos_lst:
                    current_value = position[2] + self.cost_of_cluster(position, prv_pos)
                    prv_value = self.mat[prv_pos[0] - 1][self.position_indexes[str(prv_pos[1])]]
                    if current_value < prv_value:  # solve belman equation (find min)
                        self.mat[prv_pos[0] - 1][self.position_indexes[str(prv_pos[1])]] = current_value
                        self.best_path[prv_pos[1]] = position[1]
                        # update that store is best to go from previous position to the store in parent position
                        # if cell is the origin of the problem, keep the value of the move
                        if prv_pos[0] - 1 == self.s and len(self.store_set) - 1 == self.position_indexes[str(prv_pos[1])]:
                            last_move = self.cost_of_cluster(position, prv_pos)
            final_positions = tmp_lst[:]
        self.optimal_solution = self.mat[self.s][-1]

    def restore_solution(self):
        states = []
        next_state = str([0 for i in range(self.n)])
        for i in range(self.l):
            states.append(next_state)
            if self.best_path[next_state] == str([1 for i in range(self.n)]):
                states.append(str([1 for i in range(self.n)]))
                break
            next_state = self.best_path[next_state]
        current_path = [set(self.store_set[self.position_indexes[position]]) for position in states]
        print("\nStarting From Store {}".format(self.start_store))
        print("\nClusters")
        for i in range(1, len(current_path)):  # print clusters
            print(set(current_path[i - 1]).difference(set(current_path[i])))
        print("\nOptimal Cost Is {}\n".format(self.mat[self.s][-1]))
        print(self.mat)

    def power_set(self):
        s = list(range(1, self.n + 1))
        x = len(s)
        sets = []
        for i in range(1 << x):
            sets.append([s[j] for j in range(x) if (i & (1 << j))])
        return sets

    def convert_position(self, options):
        """
        convert position stores to a bit array (store,[unvisited_bit_array],cost to get to position from current store)
        :param options:list of optional positions to go from current position (store,[stores_numbers_array])
        :return:
        """
        converted = []
        for option in options:
            converted.append((option[0], str(self.convert_to_bits(option[1])), option[2]))
        return converted

    def convert_to_bits(self, stores):
        visited_stores = [1 for i in range(self.n)]
        for store in stores:
            visited_stores[store - 1] = 0  # set the bit for every store left unvisited to 0
        return visited_stores

    def prv_positions(self, position):
        """
        generate list of possible previous positions
        given a store, find n cities forward and backwards
        for every store that you can get to,generate
        :param position: (store,unvisited)
        :return:
        """

        current_store = self.stores[position[0] - 1]
        if self.direction:
            possible_clusters = [current_store.n_forward(i) for i in range(1, self.p + 1)]
        else:
            possible_clusters = [current_store.n_back(i) for i in range(1, self.p + 1)]

        current_visited = list(map(int, position[1][1:-1].split(",")))
        added_options = set()
        optional_prv = []
        for cluster in possible_clusters:
            option = current_visited[:]
            for bit in cluster:
                option[bit - 1] = 0
            if (sum(option) == 1 and option[self.s] != 1) or option == current_visited or (
                    sum(option) == 0 and cluster[-1] != self.start_store):
                continue
            if calc_sum(position[1]) - sum(option) > self.p:
                continue
            if str(option) not in added_options:
                added_options.add(str(option))
                optional_prv.append(
                    (cluster[-1], str(option), self.mat[cluster[-1] - 1][self.position_indexes[str(option)]]))
        return optional_prv

    def cost_of_cluster(self, parent_pos, child_pos):
        return self.stores[child_pos[0] - 1].costs[abs(calc_sum(parent_pos[1]) - calc_sum(child_pos[1])) - 1]


def calc_sum(state):
    """
    count the number of visited stores
    :param state: string of state
    :return: int - sum of 1's
    """
    return sum(make_state_lst(state))


def make_state_lst(state):
    """
    convert state string to list
    :param state:
    :return:
    """
    return list(map(int, state[1:-1].split(",")))


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
    n, l, p, costs = open_file("input3.txt")
    optional_solutions = []
    for i in range(1, n+1):
        tsp1 = TSP(i, n, l, p, costs, True)
        tsp1.solve()
        optional_solutions.append(tsp1)
        tsp2 = TSP(i, n, l, p, costs, False)
        tsp2.solve()
        optional_solutions.append(tsp2)
    optimal_solution = min(optional_solutions)
    optimal_solution.restore_solution()
