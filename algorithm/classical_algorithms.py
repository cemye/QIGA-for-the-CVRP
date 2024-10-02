import numpy as np
from numpy import ndarray

from algorithm.algorithm import Algorithm
from problem_manager.problem_manager import CVRP
from helper.basic_helpers import euclidean_distance


class CWSavingsAlgorithm(Algorithm):

    def __init__(self, problem: CVRP):
        super().__init__(problem)
        self.data: ndarray = problem.data
        self.capacity_constraint: int = problem.capacity_constraint
        self.data_len: int = problem.length + 1

    def __calculate_savings__(self):
        n = self.data.shape[0]
        depot_coords = self.data[0, :2]  # Assuming first row is the depot
        savings = []
        for i in range(1, n):
            for j in range(i + 1, n):
                distance_i_to_depot = euclidean_distance(self.data[i:i + 1, :2], depot_coords[np.newaxis, :])
                distance_j_to_depot = euclidean_distance(self.data[j:j + 1, :2], depot_coords[np.newaxis, :])
                distance_i_to_j = euclidean_distance(self.data[i:i + 1, :2], self.data[j:j + 1, :2])
                s = (distance_i_to_depot + distance_j_to_depot - distance_i_to_j).item()
                savings.append(((i, j), s))
        savings.sort(key=lambda x: x[1], reverse=True)
        return savings

    def optimize(self, initial_route: list[int] = None) -> list[int]:
        savings = self.__calculate_savings__()
        routes = [[i] for i in range(1, self.data_len)]  # Initial routes
        for ((i, j), _) in savings:
            route_i = route_j = None
            for route in routes:
                if i in route:
                    route_i = route
                if j in route:
                    route_j = route
            if route_i is not route_j:
                if np.sum([self.data[k, 2] for k in route_i + route_j]) <= self.capacity_constraint:
                    routes.remove(route_i)
                    routes.remove(route_j)
                    routes.append(route_i + route_j)
        # Adding depot as start and end point
        routes = [[0] + route + [0] for route in routes]
        return routes
