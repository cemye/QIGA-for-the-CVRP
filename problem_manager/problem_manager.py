import xml.etree.ElementTree as ET
from abc import abstractmethod
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


class VehicleRoutingProblem:
    def __init__(self, filename: str):
        self.filename = filename
        self.data = self.distance_matrix = None
        self.length = 0

    def __setattr__(self, key: str, value: any) -> None:
        """
        Prevents modification of attributes after initialization.

        :param key: The attribute name.
        :param value: The value to set for the attribute.
        :raises AttributeError: If trying to set an attribute after initialization.
        """
        if getattr(self, '_initialized', False):  # Prevent changes after initial setup
            raise AttributeError("CVRP instances are immutable.")
        super().__setattr__(key, value)

    def __delattr__(self, item: str) -> None:
        """
        Prevents deletion of attributes.

        :param item: The attribute name.
        :raises AttributeError: If trying to delete an attribute.
        """
        if getattr(self, '_initialized', False):  # Prevent deleting any attribute
            raise AttributeError("CVRP instances do not support attribute deletion.")
        super().__delattr__(item)

    @abstractmethod
    def __parse_data__(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __calculate_distance_matrix__(self):
        """
        Computes the Euclidean distance matrix between all points.
        """
        coordinate_data = self.data[:, :2]
        diff = np.expand_dims(coordinate_data, axis=1) - np.expand_dims(coordinate_data, axis=0)
        self.distance_matrix = np.rint(np.sqrt(np.sum(diff ** 2, axis=-1))).astype(int)

    # -----------
    # ROUTE CALCULATIONS
    # -----------

    def calculate_path_lengths(self, routes: List[np.ndarray]) -> np.ndarray:
        """
        Calculates the total path lengths for multiple sub-routes.

        :param routes: List of sub-routes, each a numpy array of node indices.
        :return: Numpy array of path lengths for each sub-route.
        """
        return np.array([self.calculate_route_length(route) for route in routes])

    def calculate_route_length(self, route: np.ndarray) -> ndarray:
        """
        Calculates the total length of the route using the precomputed distance matrix.
        Automatically includes return to the starting point.

        :param route: Route as a numpy array of node indices.
        :return: Total route length as a float.
        """
        route_extended = np.append(route, route[0])  # Extend route to return to the start
        return np.sum(self.distance_matrix[route_extended[:-1], route_extended[1:]])

    def calculate_path_sum(self, route: List[np.ndarray]) -> float:
        """
        Calculate the sum of the requests for a given route.

        :param route: Route as a numpy array of node indices.
        :return: Sum of requests as a float.
        """
        return sum(self.calculate_path_lengths(route))


    # -----------
    # PLOTTING
    # -----------

    def plot(self, routes=None, fitness: int = None, generation: int = None):
        if not fitness and routes:
            fitness = self.calculate_path_sum(routes)

        plt.figure(figsize=(10, 10), dpi=300)  # Set the figure size for better visibility

        # Plot all nodes as scatter points
        plt.scatter(self.data[:, 0], self.data[:, 1], c="black")

        # Annotate each node with its no ID and only request
        for i in range(self.length):
            plt.annotate(f"{int(self.data[i, 2])}", (self.data[i, 0] + 1, self.data[i, 1] + 1), fontsize=9)
            # plt.annotate(f"{i}, {int(self.data[i, 2])}", (self.data[i, 0] + 1, self.data[i, 1] + 1), fontsize=9)

        if routes is not None:
            # Generate a color map for different sub-routes
            colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))

            # Plot each sub-route with a specific color
            for idx, sub_route in enumerate(routes):
                sub_route_coords = self.data[sub_route][:, :2]  # Get coordinates for the current sub-route
                plt.plot(sub_route_coords[:, 0], sub_route_coords[:, 1], marker="o", color=colors[idx], linestyle="-",
                         linewidth=2, markersize=5, label=f"Sub-route {idx + 1}")

                # Optionally, draw lines back to the depot (node 0) if needed
                depot_coords = self.data[0, :2]
                start_node_coords = self.data[sub_route[0], :2]
                end_node_coords = self.data[sub_route[-1], :2]
                plt.plot([depot_coords[0], start_node_coords[0]], [depot_coords[1], start_node_coords[1]],
                         color=colors[idx],
                         linestyle="--", linewidth=1)
                plt.plot([depot_coords[0], end_node_coords[0]], [depot_coords[1], end_node_coords[1]],
                         color=colors[idx],
                         linestyle="--", linewidth=1)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("X", fontsize=16)
        plt.ylabel("Y", fontsize=16)
        if generation is None:
            plt.title(self.filename, fontsize=16)
        else:
            plt.title(f"Routes ({'gen: ' + str(generation) + ', ' if generation else ''}fitness: {round(fitness)})")
        plt.legend()
        plt.tight_layout()
        plt.show()


class CVRP(VehicleRoutingProblem):
    def __init__(self, filename: str, cut_size: Optional[int] = None):
        """
        Initializes the Capacitated Vehicle Routing Problem instance.

        :param filename: The path to the XML file containing problem data.
        :param cut_size: Optional integer to limit the number of data points processed.
        """
        super().__init__(filename)
        self.data, self.capacity_constraint = self.__parse_data__()
        self.data = self.data[:cut_size] if cut_size is not None else self.data
        self.length = len(self.data) - 1  # Exclude the depot
        self.factorial_length = self.length - 1
        self.__calculate_distance_matrix__()

    def __parse_data__(self) -> (ndarray, float):
        tree = ET.parse(f"problem/{self.filename}.xml")
        root = tree.getroot()
        nodes_data = []
        depot_data = None  # Separate storage for depot data

        for node in root.findall(".//node"):
            cx = float(node.find("cx").text)
            cy = float(node.find("cy").text)
            request = root.find(f".//request[@node='{node.get('id')}']")
            quantity = float(request.find("quantity").text) if request is not None else 0.0
            node_data = [cx, cy, quantity]

            if node.get('type') == '0':  # Check if the node is the depot
                depot_data = node_data
            else:
                nodes_data.append(node_data)

        if depot_data is not None:
            nodes_data.insert(0, depot_data)  # Insert the depot data at the beginning of the list

        return np.array(nodes_data), float(root.find('.//vehicle_profile/capacity').text)
