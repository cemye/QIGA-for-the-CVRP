import numpy as np

from algorithm.algorithm import Algorithm


class Subroutines(Algorithm):

    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)
        self.method = self.parameters["method"]

    def optimize(self, initial_route=None, remove_depot=False):
        if self.method == "2-opt":
            return self.two_opt(initial_route)
        elif self.method == "two-opt-cvrp":
            return self.two_opt_cvrp(initial_route, remove_depot)

    def two_opt(self, subroute):
        """
        Improves a given subroute using the 2-opt algorithm.
        :param subroute: Subroute as a numpy array of node indices.
        :return: An optimized subroute as a numpy array.
        """
        best = subroute.copy()  # Copy to avoid modifying the original subroute
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):  # Adjusted to work with subroute length
                for j in range(i + 2, len(best) + (i > 0)):
                    new_route = best.copy()
                    new_route[i:j] = best[i:j][::-1]
                    if self.problem.calculate_route_length(new_route) < self.problem.calculate_route_length(best):
                        best = new_route
                        improved = True
                        break
                if improved:
                    break
        return best

    def two_opt_cvrp(self, initial_route, remove_depot):
        route = []
        for subroute in initial_route:
            route.append(np.concatenate(([0], self.two_opt(subroute[1:-1] if remove_depot else subroute), [0])))
        return route
