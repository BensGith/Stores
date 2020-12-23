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
        :param n: number of steps MUST BE SAMLLER THAN OR EQUAL N number of stores
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

    # start from solution_matrix[s,[]] - set this to 0, all other are inf
    # calculate all options and update the matrix
    # solution_matrix[s+1,[s+1]] = min(sol[s,[]] + Cs+1,1 , solution_matrix[s+1,[s+1]]) current vs prv
    # solution_matrix[s+2,[s+1,s+2]] = sol[s,[]] + Cs+2,2
    # solution_matrix[s-1,[s-1]] = sol[s,[]] + Cs-1,1
    # solution_matrix[s-2,[s-1,s-2]] = sol[s,[]] +  Cs-2,2

    # save the cells updated to queue?
    # for every cell, make a call to calculate its' options (2P OPTIONS) (1,2..p forward, 1,2..p backwards)
    # from solution_matrix[s+1,[s+1]] > solution_matrix[s+2,[s,s+1,s+2]], solution_matrix[s,[s,s+1]],  (1 and 2 forward)
    # from solution_matrix[s+1,[s+1]]

    # position_options = [store1.n_forward(i) for i in (1,p)] + [store1.n_back(i) for i in (1,p)]


    # ordered (by TSP) list of optional sets of stores left to visit
    # options = [self.stores_left[:i] for i in range(self.p_cluster_size)]

    # calculate belman's equation using the options list (2p options)
    # min{s(2,[2]) +c21,s(3,[2,3])+c32,s(2,[2,3])+c22,s(3,[3])+c31}

    # solution_matrix[self.index][self.set_index] = min(solution_matrix[self.calc_next_1()][op1],
    #                                                   solution_matrix[self.calc_next_2()][op2])


def power_set(n):
    s = list(range(1, n + 1))
    x = len(s)
    sets = []
    for i in range(1 << x):
        sets.append([s[j] for j in range(x) if (i & (1 << j))])

    return sets


def convert_position(options):
    """
    convert position stores to a bit array (store,[unvisited_bit_array],cost to get to position from current store)
    :param options:list of optional positions to go from current position (store,[stores_numbers_array])
    :return:
    """
    converted = []
    for option in options:
        converted.append((option[0], str(convert_to_bits(option[1])), option[2]))
    return converted


def convert_to_bits(stores):
    visited_stores = [1 for i in range(number_of_shops_N)]
    for store in stores:
        visited_stores[store - 1] = 0  # set the bit for every store left unvisited to 0
    return visited_stores


def prv_positions(position):
    """
    generate list of possible previous positions
    given a store, find n cities forward and backwards
    for every store that you can get to,generate
    :param position: (store,unvisited)
    :return:
    """

    current_store = store_lst[position[0] - 1]
    possible_clusters = [current_store.n_forward(i) for i in range(1, p + 1)] + [current_store.n_back(i) for i in
                                                                                 range(1, p + 1)]

    current_visited = list(map(int, position[1][1:-1].split(",")))
    added_options = set()
    optional_prv = []
    for cluster in possible_clusters:
        option = current_visited[:]
        for bit in cluster:
            option[bit - 1] = 0
        if (sum(option) == 1 and option[start_store_index] != 1) or option == current_visited or (
                sum(option) == 0 and cluster[-1] != start_store_S):
            continue
        if str(option) not in added_options:
            added_options.add(str(option))
            optional_prv.append((cluster[-1], option))
    return optional_prv


# ###### CODE START ##########
with open('input2.txt', 'r') as file:
    given_data = file.readlines()
cost_table = []
for i in range(len(given_data)):
    if i == 0:
        number_of_shops_N = int(given_data[i])
    elif i == 1:
        number_of_cars_L = int(given_data[i])
    elif i == 2:
        max_cluster_size_P = int(given_data[i])
    else:
        cost_table.append(list(map(int, given_data[i].strip().split(","))))
start_store_S = 1
start_store_index = start_store_S - 1
p = 2
s = start_store_index  # starting store index
store_sets = power_set(number_of_shops_N)
# dictionary that stores string representation of bit array -used to access the index
store_positions_indexes = {str(convert_to_bits(store_sets[i])): i for i in range(len(store_sets))}
store_indexes = {i: i - 1 for i in range(1, number_of_shops_N + 1)}
# setup matrix where all values are inf
solution_matrix = np.ones((number_of_shops_N, 2 ** number_of_shops_N)) * float('inf')
solution_matrix[s][0] = 0
store_lst = [Store(i, cost_table, max_cluster_size_P, number_of_shops_N) for i in range(1, number_of_shops_N + 1)]
# initial options to get to FINAL POSITION
options = [(store_lst[s].n_forward(i)[-1], store_lst[s].n_forward(i), store_lst[store_indexes[store_lst[s].n_forward(i)[-1]]].costs[i-1]) for i in (1, p)] + \
          [(store_lst[s].n_back(i)[-1], store_lst[s].n_back(i), store_lst[store_indexes[store_lst[s].n_back(i)[-1]]].costs[i-1]) for i in (1, p)]
final_positions = convert_position(options)


for position in final_positions:  # insert final step values to matrix (1 step to s,[]) [1,1,1] is all visited
    solution_matrix[store_indexes[position[0]]][store_positions_indexes[position[1]]] = position[2]
# iterate over paths that take 2 steps or more
# paths that take more than L cars are impossible, therefore they will be left as infinity
last_positions = final_positions
if p >= number_of_shops_N: # no need to iterate, optimal solution is to take all stores in a single cluster
    print(store_lst[s].costs[number_of_shops_N-1])
for i in range(2, number_of_cars_L+1):  # main loop
    tmp_lst = []
    for position in last_positions:
        prv_pos_lst = (prv_positions(position))  # generate 2P possible positions,remove non valid ones
        tmp_lst.append(prv_pos_lst)
        position_sum = sum(list(map(int, position[1][1:-1].split(","))))
        # for every previous position, decide if the path from it is better or not
        for prv_pos in prv_pos_lst:
            current_value = position[2] + store_lst[prv_pos[0] - 1].costs[abs(position_sum - sum(prv_pos[1]))-1]
            prv_value = solution_matrix[prv_pos[0]-1][store_positions_indexes[str(prv_pos[1])]]
            if current_value < prv_value:  # solve belman equation (find min)
                solution_matrix[prv_pos[0] - 1][store_positions_indexes[str(prv_pos[1])]] = current_value
                # update that store is best to go from previous position to the store in parent position
                # if cell is the origin of the problem, keep the value of the move
                if prv_pos[0] - 1 == s and len(store_sets) - 1 == store_positions_indexes[str(prv_pos[1])]:
                    last_move = store_lst[prv_pos[0] - 1].costs[abs(position_sum - sum(prv_pos[1]))-1]
    last_positions = tmp_lst[:]
total_cost = solution_matrix[s][-1]
current_path = [store_sets[-1]]
while total_cost > 0:
    next_move = total_cost - last_move
    state_index = np.where(solution_matrix == next_move)
    current_path.append(store_sets[state_index[1][0]])
    total_cost -= last_move
    last_move = next_move
print("\nStarting From Store {}".format(start_store_S))
print("\nClusters")
for i in range(1,len(current_path)):  # print clusters
    print(set(current_path[i-1]).difference(set(current_path[i])))
print("\nOptimal Cost Is {}\n".format(solution_matrix[s][-1]))
print(solution_matrix)


if __name__ == "__main__":
    pass
