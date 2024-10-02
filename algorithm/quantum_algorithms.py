import time

import numpy as np

from algorithm.algorithm import Algorithm
from helper.basic_helpers import leading_zero_formatter, get_str_len
from helper.bitstring_helper import bits_needed_for_permutation_as_int, segmented_sizes_for_factoradic, \
    segmented_sizes_for_vehicle_assignment
from algorithm.qiga_construct.chromosome import ClassicalChromosome
from algorithm.qiga_construct.population import Population
from problem_manager.problem_manager import CVRP


class QIGA(Algorithm):
    def __init__(self, problem: CVRP, **kwargs):
        super().__init__(problem, **kwargs)
        minimize = kwargs.get("minimize")
        enc = kwargs.get("encoding_method")
        encoding_segments = {
            "a": segmented_sizes_for_factoradic(problem.factorial_length),
            "b": segmented_sizes_for_factoradic(problem.factorial_length),
            "c": bits_needed_for_permutation_as_int(problem.factorial_length),
        }
        segments = encoding_segments.get(enc)

        ClassicalChromosome.set_class_variables(problem, enc, kwargs.get("use_gray"), segments, minimize)
        chromosome_size = int(np.sum(segments))

        self.population: Population = Population(kwargs.get("population_size"), chromosome_size,
                                                 kwargs.get("generations"), minimize, kwargs.get("simulator"))
        self.mutation_prob: float = kwargs.get("mutation_prob")
        self.rotation_init: float = kwargs.get("rotation_init")
        self.reinforcement_prob: float = kwargs.get("reinforcement_prob")
        self.rotation_min: float = kwargs.get("rotation_min", None)
        self.rotation_max: float = kwargs.get("rotation_max", None)
        self.observations = kwargs.get("observations")
        self.disaster_condition = kwargs.get("disaster_condition")

    def __do_print__(self, pop: Population, gen: int, gen_length: int, t: float):
        if gen % 50 == 0:
            print(
                f"Gen {leading_zero_formatter(gen, gen_length)}; {self.population.get_angle_diff()}; "
                f"{pop.generation_best_chromosome.idx}: {round(pop.generation_best_chromosome.fitness)}; t:{np.round(time.time() - t, 3)}s")

    def __do_inits__(self, use_random_init: bool = False):
        t = time.time()
        pop = self.population
        gen_length = get_str_len(pop.GENERATIONS)

        #  Quantum Initialization Circuit
        print(f"Gen {leading_zero_formatter(0, gen_length)}")
        if pop.quantum_init_rotation != 0:
            pop.quantum_init_random() if use_random_init else pop.quantum_init_rotation(self.rotation_init)
        pop.observe_and_evaluate(self, 0)
        return t, pop, gen_length

    def optimize(self, initial_route: list[int] = None) -> Population:
        t, pop, gen_length = self.__do_inits__()
        for gen in range(1, pop.GENERATIONS + 1):
            self.__do_print__(pop, gen, gen_length, t)
            t = time.time()
            angle = pop.get_self_adaptive_magnitude_linear(self.rotation_min, self.rotation_max, gen) * np.pi
            for chrom in pop.generation_best_quantum_chromosomes:
                pop.reinforcement(chrom.idx, pop.global_best_chromosome, angle)
            pop.q_zero_mutation_all(self.mutation_prob)
            pop.observe_and_evaluate(self, gen)
        return pop
