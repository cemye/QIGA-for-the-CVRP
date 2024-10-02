from typing import Callable

import numpy as np
from numpy import ndarray

from algorithm.subroutines import Subroutines
from helper.bitstring_helper import array_to_bitstring, repair_factoradic_with_modulo, factoradic_to_permutation, \
    bitstring_to_int_list, bitstring_to_single_int, single_int_to_factoradic, repair_factoradic_with_scaling
from problem_manager.problem_manager import CVRP


class ClassicalChromosome:
    cvrp: CVRP = None
    use_gray: bool = True
    minimize: bool = True
    segments: ndarray = None
    max_segment_sizes: ndarray = None
    decode_method: Callable[[], None] = None

    @classmethod
    def set_class_variables(cls, cvrp_instance, encoding, use_gray, segments, minimize):
        cls.cvrp = cvrp_instance
        cls.use_gray = use_gray
        cls.segments = segments
        cls.minimize = minimize
        cls.max_segment_sizes = np.arange(cls.cvrp.factorial_length, 0, -1)
        encodings = {
            "a": cls.decode_bitstring_a,
            "b": cls.decode_bitstring_b,
            "c": cls.decode_bitstring_c,
            "d": cls.decode_bitstring_d
        }
        if encoding not in encodings:
            raise ValueError(f"Invalid encoding: '{encoding}'. Valid options are: {list(encodings.keys())}.")
        cls.decode_method = encodings.get(encoding)

    def __init__(self, bitstring: ndarray, idx: int, generation: int,
                 route: list | ndarray = None, fitness: int | float = None):
        self.bitstring: ndarray = bitstring
        self.idx: int = idx
        self.generation: int = generation
        self.route: list | ndarray = route
        self.fitness: int | float = fitness

    def __str__(self):
        return f"{self.get_bitstring}, f: {self.fitness}"

    def __repr__(self):
        return f"id: {self.idx}, f: {self.fitness}"
        # return f"id: {self.idx}, f: {self.fitness}, {self.bitstring}, {self.route}"

    def is_better_than(self, other):
        """Returns True if this chromosome is better than the 'other' based on the fitness comparison."""
        return self.fitness < other.fitness if self.minimize else self.fitness > other.fitness

    def get_bitstring(self):
        return array_to_bitstring(self.bitstring)

    def get_bitstring_distribution(self):
        ones = np.sum(self.bitstring)
        return ones, len(self.bitstring) - ones

    def calculate_fitness(self):
        self.decode_method()
        self.fitness = self.cvrp.calculate_path_sum(self.route)

    def decode_bitstring_a(self):
        factoradic = bitstring_to_int_list(self.bitstring, self.segments, self.use_gray)
        factoradic = repair_factoradic_with_modulo(factoradic)
        self.route = factoradic_to_permutation(factoradic) + 1
        self.basic_split()
        self.route = Subroutines(self.cvrp, method="two-opt-cvrp").optimize(self.route, True)

    def decode_bitstring_b(self):
        factoradic = bitstring_to_int_list(self.bitstring, self.segments, self.use_gray)
        factoradic = repair_factoradic_with_scaling(factoradic, self.segments, self.max_segment_sizes)
        self.route = factoradic_to_permutation(factoradic) + 1
        self.basic_split()
        self.route = Subroutines(self.cvrp, method="two-opt-cvrp").optimize(self.route, True)

    def decode_bitstring_c(self):
        single_int = bitstring_to_single_int(self.bitstring, self.use_gray)
        factoradic = single_int_to_factoradic(single_int, self.cvrp.factorial_length)
        self.route = factoradic_to_permutation(factoradic) + 1
        self.basic_split()
        self.route = Subroutines(self.cvrp, method="two-opt-cvrp").optimize(self.route, True)

    def decode_bitstring_d(self):
        customers = bitstring_to_int_list(self.bitstring, self.segments, self.use_gray)
        self.route = np.argsort(customers) + 1  # Sort customers based on their index
        self.basic_split()
        self.route = Subroutines(self.cvrp, method="two-opt-cvrp").optimize(self.route, True)

    def plot_route(self):
        self.cvrp.plot(self.route, self.fitness, self.generation)

    def greedy_split(self):
        # Start at the depot (index 0 in most cases)
        routes = []
        current_route = [0]  # Start the route with the depot
        current_load = 0
        for node in self.route:
            if node == 0:
                continue  # Skip depot node in tour
            demand = self.cvrp.data[node, 2]

            if current_load + demand <= self.cvrp.capacity_constraint:
                current_route.append(node)
                current_load += demand
            else:
                # Finalize current route by adding depot at the end and start a new one
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, node]  # Start new route with depot and current node
                current_load = demand

        # Don't forget to add the last route if it exists
        if current_route != [0]:
            current_route.append(0)  # End the last route with depot
            routes.append(current_route)

        self.route = routes

    def basic_split(self):
        predecessor_list = split_route(self.cvrp, self.route)
        self.route = extract_routes(self.route, predecessor_list)


# -----------------
# Splits
# -----------------

def split_route(cvrp: CVRP, global_route: list[int]) -> list[list[int]]:
    # Initialize the depot as the 0th node
    depot = 0
    dm = cvrp.distance_matrix
    demands = [item[2] for item in cvrp.data]  # Extracting the demand from the data
    n = len(global_route)

    # Variables to store the results
    V = [float('inf')] * (n + 1)
    P = [-1] * (n + 1)
    V[0] = 0  # Initial cost is zero since we're starting at the depot

    # Outer loop over each possible starting point for a route
    for i in range(1, n + 1):
        load = 0
        j = i
        while j <= n and load + demands[global_route[j - 1]] <= cvrp.capacity_constraint:
            load += demands[global_route[j - 1]]
            if i == j:  # Starting a new route from i to j (same)
                cost = dm[depot][global_route[i - 1]] + dm[global_route[i - 1]][depot]
            else:  # Extending the current route
                cost = (dm[global_route[j - 2]][global_route[j - 1]]
                        - dm[depot][global_route[j - 2]]
                        + dm[depot][global_route[j - 1]])

            if load <= cvrp.capacity_constraint and V[i - 1] + cost < V[j]:
                V[j] = V[i - 1] + cost
                P[j] = i - 1  # Track the split point
            j += 1

    return P


def extract_routes(global_route: list[int], predecessor_list: list[int]) -> list[list[int]]:
    routes = []
    current_index = len(global_route)  # Start from the last customer
    while current_index > 0:
        previous_index = predecessor_list[current_index]
        route_segment = global_route[previous_index:current_index].tolist()
        route = [0] + route_segment + [0]  # Insert the depot at the start and end
        routes.append(route)
        # Move to the predecessor
        current_index = previous_index

    # Reverse routes to get them in correct order
    return routes[::-1]  # Since routes are constructed backwards
