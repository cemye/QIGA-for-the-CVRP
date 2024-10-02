# Quantum Inspired Genetic Algorithm for CVRP

This project implements a Quantum-inspired Genetic Algorithm (QIGA) to solve the Capacitated Vehicle Routing Problem (
CVRP). The algorithm leverages the superposition principle to optimize routes for a fleet of vehicles delivering goods
to a set of customers with specific demands.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithm Details](#algorithm-details)


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/cemye/qiga-for-the-cvrp.git
    cd qiga-cvrp
    ```

2. Create a Conda environment from the `environment.yml` file. This process may take a while:
    ```sh
    conda env create -f environment.yml
    ```

3. Activate the Conda environment:
    ```sh
    conda activate qiga-env
    ```

## Usage

To run the QIGA for CVRP, you need to provide the problem instance and configure the algorithm parameters. Here is an
example of how to use the algorithm:

```python
from algorithm.quantum_algorithms import QIGA
from problem_manager.problem_manager import CVRP

# Load your CVRP problem instance from the /problem folder
problem = CVRP("CMT01")

# Initialize the QGA with the problem and parameters
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

# Optimize the route
population = QIGA(problem, **qga_config).optimize()
best_route = population.global_best_chromosome.route

print("Best route found:", best_route)
```
An implementation of this example is given in `bench.py`.

## Project Structure

- `algorithm/`: Contains the implementation of the QIGA and related classes.
    - `qga_construct/`: Contains the core QGA implementation.
- `analysis_and_figures/`: Contains scripts and notebooks for analyzing encodings and theresults of the algorithm.

- `helper/`: Contains utility functions for bitstring manipulation, quantum operations and other helper methods.
- `problem_manager/`: Contains classes and methods for managing CVRP problem instances.
- `problem/`: Contains CVRP problem instances.
- `bench.py`: Script for running benchmarks and testing the QIGA implementation.

## Algorithm Details

The QIGA combines classical genetic algorithm principles with quantum computing techniques to
enhance the search for optimal solutions. Key components include:

- **Quantum Initialization**: Initializes the population using quantum circuits.
- **Evolutionary Loop**:
    - **Evolutionary Operations**: Applies quantum evolutionary operations to evolve the population. (including
      selection, quantum reinforcement, quantum rotation, crossover and mutation)
    - **Observation & Evaluation**: Measures the population to generate the classical population and to evaluate the
      fitness of each classical chromosome.
  - **Disaster**: Resets part of the population if it gets stuck in local optima for too long.

