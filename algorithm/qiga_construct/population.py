import heapq
from typing import Any, Generator

import numpy as np
from numpy import ndarray
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from helper.basic_helpers import leading_zero_formatter, normalize_angles
from helper.bitstring_helper import split_results_into_chromosome_chunks_with_shots
from algorithm.qiga_construct.chromosome import ClassicalChromosome
from algorithm.qiga_construct.progress_data import ProgressData
from helper.quantum_helpers import generate_q_registers, simulate


class Population:

    def __init__(self, population_size: int, chromosome_size: int, generations: int, minimize: bool, simulator: str):
        self.GENERATIONS: int = generations
        self.CHROMOSOME_SIZE: int = chromosome_size
        self.SIZE: int = population_size
        self.QUBIT_COUNT: int = self.SIZE * self.CHROMOSOME_SIZE
        self.QUBIT_COUNT_LENGTH: int = len(str(self.QUBIT_COUNT))  # string length
        self.minimize: bool = minimize

        self.q_angles: ndarray = np.zeros((population_size, chromosome_size), dtype=float)
        self.q_register: list[QuantumRegister] = generate_q_registers(population_size, chromosome_size)
        self.qc: QuantumCircuit = QuantumCircuit(*self.q_register)
        self.theta_vector = [ParameterVector(f"G{i}_theta", chromosome_size) for i in range(population_size)]
        self.construct_qc()  # apply gates and theta_vector
        self.data: ProgressData = ProgressData(self.SIZE, self.CHROMOSOME_SIZE, self.GENERATIONS)
        self.classical_population: list[ClassicalChromosome] = []
        self.global_best_chromosome: ClassicalChromosome | None = None
        self.generation_best_chromosome: ClassicalChromosome | None = None
        self.best_quantum_chromosomes: {int: ClassicalChromosome} = {}
        self.generation_best_quantum_chromosomes: list[ClassicalChromosome] = []
        self.local_optima_duration: int = 0

        self.rng = np.random.default_rng()
        self.simulator = AerSimulator(method=simulator)

    def construct_qc(self):
        for reg in self.q_register:
            self.qc.h(reg)

        for idx in range(self.SIZE):
            for idq in range(self.CHROMOSOME_SIZE):
                self.qc.ry(self.theta_vector[idx][idq], self.q_register[idx][idq])
        self.qc.measure_all()

    def get_angle_diff(self):
        flattened_arr = self.q_angles.flatten()
        return leading_zero_formatter(np.sum(flattened_arr > 0) - np.sum(flattened_arr < 0), self.QUBIT_COUNT_LENGTH)

    def get_non_best_chromosome(self, number: int, worst: bool = True) -> list[ClassicalChromosome]:
        """
        Get non-best chromosomes from the classical population.
        Parameters:
            number (int): The number of non-best chromosomes to retrieve. (-1 for all except the best)
            worst (bool): Flag indicating whether to select the worst or best chromosomes. Defaults to worst.
        Returns:
            list: List of non-best classical chromosomes.
        """
        non_best = self.generation_best_quantum_chromosomes[1:]
        if number == -1:
            return non_best[::-1] if worst else non_best
        return non_best[-number:] if worst else non_best[:number]

    def get_worst_chromosome(self, number: int = 1) -> list[ClassicalChromosome]:
        """
        Get the worst chromosome from the population.
        Parameters:
            number (int): The number of worst chromosomes to retrieve. Defaults to 1.
        Returns:
            list: List of worst chromosomes.
        """
        return self.get_non_best_chromosome(number, worst=True)

    # --------------------------------
    # INIT CIRCUIT
    # --------------------------------
    def quantum_init_rotation(self, initial_rotation: float = 0):
        # set q angles with random sign and initial rotation
        np.multiply(self.rng.choice([-1, 1], size=self.q_angles.shape), initial_rotation * np.pi, out=self.q_angles)

    def quantum_init_random(self):
        self.q_angles = self.rng.uniform(low=-np.pi, high=np.pi, size=self.q_angles.shape)

    # --------------------------------
    # OBSERVATION & EVALUATION
    # --------------------------------

    def observe_and_evaluate(self, algorithm, generation: int):
        self.observe_q_chromosome(generation, algorithm.observations)
        self.evaluate_population(algorithm, generation)

    def observe_q_chromosome(self, generation: int, shots: int):
        circuit = self.qc.assign_parameters(self.q_angles.flatten())
        bitstring_lists = split_results_into_chromosome_chunks_with_shots(self.SIZE, self.QUBIT_COUNT,
                                                                          simulate(circuit, self.simulator, shots))
        self.classical_population = [ClassicalChromosome(bitstring, idx, generation) for idx, bitstring_list in
                                     enumerate(bitstring_lists) for bitstring in bitstring_list]

    def evaluate_population(self, algorithm, generation: int):
        # Step 1: Calculate fitness for all solutions
        for sol in self.classical_population:
            sol.calculate_fitness()

        # Step 2: Identify the best chromosomes for this generation
        best_chromosomes = {}
        for chromosome in self.classical_population:
            idx = chromosome.idx
            current_best = best_chromosomes.get(idx)
            if current_best is None or chromosome.is_better_than(current_best):
                best_chromosomes[idx] = chromosome

        # Step 3: Store the best chromosomes sorted by fitness
        self.generation_best_quantum_chromosomes = sorted(best_chromosomes.values(), key=lambda x: x.fitness,
                                                          reverse=not self.minimize)

        # Step 4: Update global best for each quantum chromosomes
        if generation == 0:
            self.best_quantum_chromosomes = best_chromosomes
        else:
            for idx, chrom in best_chromosomes.items():
                existing_best = self.best_quantum_chromosomes.get(idx)
                if chrom.is_better_than(existing_best):
                    self.best_quantum_chromosomes[idx] = chrom

        # Step 5: Identify the local best chromosome
        self.generation_best_chromosome = self.generation_best_quantum_chromosomes[0]

        # Step 6: Update global best chromosome if the current best is better
        update_condition = generation == 0 or self.generation_best_chromosome.is_better_than(
            self.global_best_chromosome)

        if update_condition:
            self.global_best_chromosome = self.generation_best_chromosome
            self.local_optima_duration = 0
        else:
            self.local_optima_duration += 1

        # Step 7: Update algorithm's data tracking
        self.data.update_data(self, generation, update_condition)

    # ----------------------------------------
    # Quantum Rotation Gate / Reinforcement
    # ----------------------------------------

    def reinforcement(self, to_reinforce_idx: int, goal: ClassicalChromosome, reinforcement: float,
                      probability: float = 1, deterministic: bool = False):
        idx = to_reinforce_idx
        mutation_mask = self.rng.random(self.q_angles[idx].shape) < probability

        def reinforce_angles():
            reinforcement_values = np.where(goal.bitstring, reinforcement, -reinforcement)
            reinforced_angles = self.q_angles[idx] + mutation_mask * reinforcement_values
            self.q_angles[idx] = normalize_angles(reinforced_angles)

        def deterministic_q_elitism():
            self.q_angles[idx] = np.where(mutation_mask & (goal.bitstring == "1"), np.pi / 2, self.q_angles[idx])

        deterministic_q_elitism() if deterministic else reinforce_angles()

    def q_rotation_gate(self, to_rotate: ClassicalChromosome, goal: ClassicalChromosome, angle: float):
        sign_mask = np.where(goal.bitstring > to_rotate.bitstring, 1,
                             np.where(goal.bitstring < to_rotate.bitstring, -1, 0))
        self.q_angles[to_rotate.idx] += angle * sign_mask
        self.q_angles[to_rotate.idx] = normalize_angles(self.q_angles[to_rotate.idx])

    def q_rotation_angle_gate(self, idx_to_rotate: int, best_idx: int, angle: float,
                              inplace: bool = True) -> None | ndarray:

        sign_mask = np.where(self.q_angles[best_idx] < self.q_angles[idx_to_rotate], 1,
                             np.where(self.q_angles[best_idx] > self.q_angles[idx_to_rotate], -1,
                                      (-1) ** self.rng.integers(2)))
        angle_adjustments = angle * sign_mask

        if inplace:
            self.q_angles[idx_to_rotate] += angle_adjustments
            self.q_angles[idx_to_rotate] = normalize_angles(self.q_angles[idx_to_rotate])
        else:
            adjusted_angles = self.q_angles[idx_to_rotate] + angle_adjustments
            return normalize_angles(adjusted_angles)

    def q_rotation_angle_gate_all(self, best_idx: int, angle: float, skip_best: bool = False,
                                  inplace: bool = True) -> None | ndarray:
        sign_mask = np.where(self.q_angles < self.q_angles[best_idx], 1,
                             np.where(self.q_angles > self.q_angles[best_idx], -1,
                                      (-1) ** np.random.randint(2, size=self.q_angles.shape)))

        if skip_best:
            sign_mask[best_idx] = 0
        angle_adjustments = angle * sign_mask

        if inplace:
            self.q_angles += angle_adjustments
            self.q_angles = normalize_angles(self.q_angles)
            return
        adjusted_angles = self.q_angles + angle_adjustments
        return normalize_angles(adjusted_angles)

    def get_self_adaptive_magnitude_exponential(self, angle: float, gen: int) -> float:
        return angle * np.exp(-gen / self.GENERATIONS)

    def get_self_adaptive_magnitude_exponential2(self, angle_min: float, scale: float, gen: int) -> float:
        return angle_min + (1 - angle_min) * np.exp(-gen * scale / self.GENERATIONS)

    def get_self_adaptive_magnitude_linear(self, angle_min: float, angle_max: float, gen: int) -> float:
        return angle_max - (angle_max - angle_min) * gen / self.GENERATIONS

    def get_self_adaptive_observations(self, obs_min: int, obs_max: int, gen: int, steepness_factor: int = 5) -> int:
        return int(np.floor(obs_min + (obs_max - obs_min) * np.exp(-steepness_factor * gen / self.GENERATIONS)))

    # --------------------------------
    # SELECTION
    # --------------------------------
    def best_q_chroms_selection(self, return_ids: bool = False) -> list[int] | list[Any] | Any:
        """
        Selects the best classical chromosome for each quantum chromosome.

        Returns:
            A sorted list of the best chromosomes by fitness; the best fitness first.
        """
        best_chromosomes = {}
        for chromosome in self.classical_population:
            idx = chromosome.idx
            if idx not in best_chromosomes or chromosome.fitness > best_chromosomes[idx].fitness:
                best_chromosomes[idx] = chromosome

        if return_ids:
            return [chromosome.idx for chromosome in sorted(best_chromosomes.values(), key=lambda x: x.fitness)]
        else:
            return sorted(best_chromosomes.values(), key=lambda x: x.fitness)

    def roulette_wheel_selection(self, chromosome_list: list[ClassicalChromosome], num_selections: int) \
            -> list[ClassicalChromosome] | ClassicalChromosome:
        fitnesses = np.array([chromosome.fitness for chromosome in chromosome_list])
        total_fitness = np.sum(fitnesses)
        running_sums = np.cumsum(fitnesses)
        selections = []
        for _ in range(num_selections):
            selection_point = self.rng.uniform(0, total_fitness)

            # Find the index of the first chromosome that makes the running sum exceed the selection point
            index = np.nonzero(running_sums > selection_point)[0][0]
            selections.append(chromosome_list[index])
        return selections[0] if len(selections) == 1 else selections

    def stochastic_universal_sampling_selection(self, chromosome_list: list[ClassicalChromosome], num_selections: int):
        fitnesses = np.array([chromosome.fitness for chromosome in chromosome_list])
        total_fitness = np.sum(fitnesses)
        distance = total_fitness / num_selections
        start_point = self.rng.uniform(0, distance)
        points = start_point + np.arange(num_selections) * distance

        running_sum = np.cumsum(fitnesses)
        indices = np.searchsorted(running_sum, points)

        return [chromosome_list[index] for index in indices]

    def tournament_selection(self, chromosome_list: list[ClassicalChromosome], num_selections: int,
                             tournament_size: int = 3):
        selections = []
        for _ in range(num_selections):
            indices = self.rng.integers(0, len(chromosome_list), tournament_size)
            tournament = np.array([chromosome_list[i].fitness for i in indices])
            winner_idx = indices[np.argmin(tournament) if self.minimize else np.argmax(tournament)]
            selections.append(chromosome_list[winner_idx])
        return selections

    # --------------------------------
    # CROSSOVER
    # --------------------------------

    def one_point_crossover(self, idx1: int, idx2: int) -> list[np.ndarray]:
        crossover_point = self.rng.integers(1, self.CHROMOSOME_SIZE - 1)
        x1, x2 = self.q_angles[idx1], self.q_angles[idx2]
        return [np.r_[x1[:crossover_point], x2[crossover_point:]],
                np.r_[x2[:crossover_point], x1[crossover_point:]]]

    def n_point_crossover(self, parent1: ClassicalChromosome, parent2: ClassicalChromosome, num_points: int):
        crossover_points = self.rng.choice(range(1, self.CHROMOSOME_SIZE - 1), num_points, replace=False)
        crossover_points.sort()

        x1, x2 = self.q_angles[parent1.idx], self.q_angles[parent2.idx]
        children = [x1]
        for i in range(num_points):
            children.append(np.r_[children[-1][:crossover_points[i]], x2[crossover_points[i]:]])
            children.append(np.r_[children[-1][crossover_points[-1]:]])

        return children

    def chaos_crossover(self, idx1: int, crossover_point: int = None) -> list[np.ndarray]:
        crossover_point = self.CHROMOSOME_SIZE // 2 if crossover_point is None else crossover_point
        x = self.q_angles[idx1]
        return [np.r_[x[:crossover_point], np.zeros_like(self.q_angles[idx1][crossover_point:], dtype=float)],
                np.r_[np.zeros_like(self.q_angles[idx1][:crossover_point], dtype=float), x[crossover_point:]]]

    def full_chaos_crossover(self, idx: int, num_children: int) -> np.ndarray:
        parent = self.q_angles[idx]
        segment_length = self.CHROMOSOME_SIZE // num_children
        children = np.zeros((num_children, self.CHROMOSOME_SIZE))

        for i in range(num_children):
            start = i * segment_length
            end = start + segment_length if i < num_children - 1 else self.SIZE
            children[i, start:end] = parent[start:end]

        return children

    def whole_arithmetic_crossover(self, idx1: int, idx2: int, alpha: int = 0.5) -> np.ndarray | list[np.ndarray]:
        first_child = alpha * self.q_angles[idx1] + (1 - alpha) * self.q_angles[idx2]
        if alpha == 0.5:
            return first_child
        return [first_child, alpha * self.q_angles[idx2] + (1 - alpha) * self.q_angles[idx1]]

    def addition_crossover(self, idx1: int, idx2: int, magnitude1: int = 0.5, magnitude2: int = 0.5,
                           normalize: bool = True):
        """Update q_angles at idx1 and idx2 to their sum"""
        angles = magnitude1 * self.q_angles[idx1] + magnitude2 * self.q_angles[idx2]
        return normalize_angles(angles) if normalize else angles

    def interference_crossover(self, classical_chromosomes: list[ClassicalChromosome]) -> np.ndarray:
        """
            Calculate the interference crossover values for classical chromosomes.

            Parameters:
                self: object
                    The current object instance.
                classical_chromosomes: list
                    A list of classical chromosomes, each represented as an object with a 'bitstring' attribute.

            Returns:
                numpy.ndarray
                    An array containing the interference crossover values for each position in the bitstrings of the classical chromosomes
            """
        bitstring_matrix = np.array([[int(bit) for bit in obj.bitstring] for obj in classical_chromosomes])
        sums = np.sum(bitstring_matrix, axis=0)
        max_value = len(classical_chromosomes)
        return ((sums / max_value) * np.pi) - (np.pi / 2)

    def q_interference_crossover(self):
        children = np.empty_like(self.q_angles)
        interference = (np.arange(self.SIZE).reshape(-1, 1) + np.arange(
            self.CHROMOSOME_SIZE)) % self.SIZE

        for col_idx in range(self.CHROMOSOME_SIZE):
            children[:, col_idx] = self.q_angles[interference[:, col_idx], col_idx]

        return children

    # --------------------------------
    # MUTATION
    # --------------------------------

    def q_inversion_mutation(self, idx: int, probability: float):
        if probability != 0:
            mutation_mask = self.rng.random(self.q_angles[idx].shape) < probability
            self.q_angles[idx] *= np.where(mutation_mask, -1, 1)

    def q_zero_mutation(self, idx: int, probability: float):
        if probability != 0:
            mutation_mask = self.rng.random(self.q_angles[idx].shape) < probability
            self.q_angles[idx] *= np.where(mutation_mask, 0, 1)
        elif probability == 1:
            self.q_angles[idx] = np.zeros(self.CHROMOSOME_SIZE, dtype=float)

    def q_inversion_mutation_all(self, probability: float):
        if probability != 0:
            mutation_mask = self.rng.random(self.q_angles.shape) < probability
            self.q_angles *= np.where(mutation_mask, -1, 1)

    def q_zero_mutation_all(self, probability: float):
        if probability != 0:
            mutation_mask = self.rng.random(self.q_angles.shape) < probability
            self.q_angles *= np.where(mutation_mask, 0, 1)
        elif probability == 1:
            self.q_angles = np.zeros(self.q_angles.shape, dtype=float)

    def q_swap_mutation(self, idx: int, probability: float):
        chromosome = self.q_angles[idx]
        swap_decisions = np.random.rand(self.CHROMOSOME_SIZE) < probability
        indices = np.where(swap_decisions)[0]  # Get indices of the genes to be swapped
        np.random.shuffle(indices)

        # Swap the elements based on the shuffled indices
        for i in range(0, len(indices) - 1, 2):
            chromosome[indices[i]], chromosome[indices[i + 1]] = chromosome[indices[i + 1]], chromosome[indices[i]]

        self.q_angles[idx] = chromosome
