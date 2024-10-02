import time
from algorithm.quantum_algorithms import QIGA
from helper.basic_helpers import convert_to_printable_time
from problem_manager.problem_manager import CVRP

# Hyperparameters
qga_config = {
    "population_size": 10,
    "generations": 200,

    # Encoding
    "encoding_method": "b",  # a, b, c
    "use_gray": False,
    "minimize": True,

    # Simulation
    "simulator": "matrix_product_state",
    "observations": 20,

    # Evolutionary Operation
    "rotation_init": 1 / 8,
    "reinforcement_prob": 1,
    "rotation_max": 0.25,
    "rotation_min": 0.01,
    "mutation_prob": 0.005,
}

problem = CVRP("CMT01")
start = time.time()
population = QIGA(problem, **qga_config).optimize()
print(f"Time: {convert_to_printable_time(time.time() - start)}")
population.global_best_chromosome.plot_route()
population.data.plot()
