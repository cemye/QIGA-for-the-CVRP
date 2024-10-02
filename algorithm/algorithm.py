from abc import abstractmethod

from problem_manager.problem_manager import VehicleRoutingProblem


class Algorithm:
    def __init__(self, problem: VehicleRoutingProblem, **kwargs):
        self.problem: VehicleRoutingProblem = problem
        self.parameters = kwargs

    @abstractmethod
    def optimize(self, initial_route=None):
        raise NotImplementedError("The method 'run' is not implemented in the base class.")
