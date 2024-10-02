from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from helper.bitstring_helper import average_hamming_distance

dBlue = "#003f6b"
lBlue = "#00abda"
orange = "#d53d0e"
plt.rcParams['figure.dpi'] = 300


class ProgressData:
    def __init__(self, population_size: int, chromosome_size: int, generations: int):
        self.generations = generations
        self.generations_plus = generations + 1
        self.population_size = population_size
        self.q_angles_generational = np.zeros((generations + 1, population_size, chromosome_size))
        self.best_local_per_chromosome = {}
        self.global_best_solutions = []
        self.q_angle_diversity = []
        self.classical_chromosomes = []
        self.fitness_values = []
        self.fitness_stats = []
        self.q_angle_mean_abs = np.zeros((generations + 1, population_size))
        self.q_angle_count_free = np.zeros((generations + 1, population_size))

    def plot(self):
        self.calculations()
        self.plot_fitness_boxplot()
        self.plot_fitness_scatterplot()
        self.plot_global_best_solutions()
        self.plot_population_diversity()
        self.plot_q_angle_range()
        self.plot_q_angle_count_free()
        self.plot_fitness_over_generations_per_quantum_chromosome()

    def update_data(self, pop, generation: int, update_global_best: bool) -> None:
        self.q_angles_generational[generation, :, :] = pop.q_angles.copy()
        self.classical_chromosomes.append(pop.classical_population)
        self.best_local_per_chromosome[generation] = pop.best_quantum_chromosomes.copy()

        if update_global_best:
            pop.data.global_best_solutions.append(pop.global_best_chromosome)

    def calculations(self):
        for gen, angles in enumerate(self.q_angles_generational):
            self.q_angle_diversity.append(np.mean(np.std(angles, axis=0)))
            self.q_angle_mean_abs[gen, :] = (np.mean(np.abs(angles), axis=1))
            self.q_angle_count_free[gen, :] = (np.sum((angles >= -1) & (angles <= 1), axis=1) / angles.shape[1])

        for solutions in self.classical_chromosomes:
            fits = np.array([chromo.fitness for chromo in solutions])
            self.fitness_values.append(fits)
            self.fitness_stats.append({
                'mean': np.mean(fits),
                'std': np.std(fits),
                'max': np.max(fits),
                'min': np.min(fits),
                'median': np.median(fits)})

    def plot_q_angle_range(self):
        transposed_data = self.q_angle_mean_abs.T
        plt.figure(figsize=(8, 6))
        x_values = np.arange(self.generations_plus)
        for index, element_data in enumerate(transposed_data):
            plt.plot(x_values, element_data, label=f'Chromosome {index}')
        plt.axhline(y=np.pi / 2, color=orange, linestyle='--', label=r'$Range Limit$')
        plt.xlabel('Generation')
        plt.ylabel('Radian')
        # plt.title('Mean Absolute Angle by Chromosome')
        plt.xlim(left=0, right=self.generations)
        plt.ylim(bottom=0)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_q_angle_count_free(self):
        transposed_data = self.q_angle_count_free.T
        plt.figure(figsize=(8, 6))
        x_values = np.arange(self.generations_plus)
        for index, element_data in enumerate(transposed_data):
            plt.plot(x_values, element_data, label=f'Element {index}')
        plt.xlabel('Generation')
        plt.ylabel('Percentage')
        # plt.title('Count of Angles within [-1, 1] by Chromosome')

        plt.xlim(left=0, right=self.generations)
        plt.ylim(bottom=0)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_population_diversity(self):
        average_distances = [average_hamming_distance(genes) for genes in self.classical_chromosomes]

        fig, ax1 = plt.subplots(figsize=(8, 6))

        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Mean Standard Deviation', color=dBlue)
        ax1.plot(self.q_angle_diversity, color=dBlue, label='Quantum Angle Diversity')
        ax1.tick_params(axis='y', labelcolor=dBlue)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Average Hamming Distance', color=lBlue)
        ax2.plot(range(self.generations_plus), average_distances, color=lBlue,
                 label='Bitstring Diversity')
        ax2.tick_params(axis='y', labelcolor=lBlue)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.title('Population Diversity')
        plt.grid(True)
        plt.xlim(left=0, right=self.generations)
        plt.ylim(bottom=0)

        # Adding legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.show()

    def plot_q_diversity(self):
        plt.plot(self.q_angle_diversity, color=lBlue)
        plt.figure(figsize=(12, 6))
        # plt.title('Quantum Angle Diversity')
        plt.xlabel('Generation')
        plt.ylabel('Mean Standard Deviation')
        plt.grid(True)
        plt.show()

    def plot_fitness_boxplot(self):
        plt.figure(figsize=(18, 9))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        generations = np.arange(self.generations_plus)
        # Generating box plot directly at generation indices
        bp = plt.boxplot(self.fitness_values, positions=generations, widths=0.6, notch=True, showfliers=False,
                         patch_artist=True)

        # Adjusting line properties manually
        for element in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
            plt.setp(bp[element], linewidth=1.0)
        for box in bp['boxes']:
            box.set(facecolor='lightgray', alpha=0.6)

        max_values = [stat['max'] for stat in self.fitness_stats]
        min_values = [stat['min'] for stat in self.fitness_stats]
        median_values = [stat['median'] for stat in self.fitness_stats]

        plt.plot(generations, max_values, '--', label='Max Fitness', color='red', alpha=0.5)
        plt.plot(generations, min_values, '--', label='Min Fitness', color='blue', alpha=0.5)
        # plt.plot(generations, average_values, '--', label='Average Fitness', color='orange', alpha=0.5)
        plt.plot(generations, median_values, '-', label='Median Fitness', color='orange', alpha=0.5)

        # Dynamically adjust the x-axis ticks to ensure there are about 20 ticks
        step_size = max(self.generations_plus // 20, 1)
        ticks = np.arange(generations[0], generations[-1] + 1, step_size)
        plt.xticks(ticks, [str(gen) for gen in ticks], rotation=45)

        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        # plt.title("Fitness Distribution over Generations", fontsize=14)

        # Adjust x-limits to snugly fit the range of generations
        plt.xlim(left=generations[0] - 0.5, right=generations[-1] + 0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_fitness_scatterplot(self):
        plt.figure(figsize=(18, 9))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        generations = np.arange(self.generations_plus)

        # Plot each individual fitness data point
        for gen_index, fitness_values in enumerate(self.fitness_values):
            x_values = [gen_index] * len(fitness_values)  # Create an x-value for each fitness point in this generation
            plt.scatter(x_values, fitness_values, alpha=0.6, edgecolors='none',
                        s=10)  # s controls the size of the scatter points

        # Calculate and plot lines for max, min, and median values
        max_values = [stat['max'] for stat in self.fitness_stats]
        min_values = [stat['min'] for stat in self.fitness_stats]
        median_values = [stat['median'] for stat in self.fitness_stats]

        plt.plot(generations, max_values, '--', label='Max Fitness', color='red', alpha=0.5)
        plt.plot(generations, min_values, '--', label='Min Fitness', color='blue', alpha=0.5)
        plt.plot(generations, median_values, '-', label='Median Fitness', color='orange', alpha=0.5)

        # Set ticks and labels
        step_size = max(self.generations_plus // 20, 1)
        ticks = np.arange(generations[0], generations[-1] + 1, step_size)
        plt.xticks(ticks, [str(gen) for gen in ticks], rotation=45)

        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        # plt.title("Fitness Distribution over Generations", fontsize=14)

        # Adjust x-limits to snugly fit the range of generations
        plt.xlim(left=generations[0] - 0.5, right=generations[-1] + 0.5)

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_hamming_diversity(self):
        average_distances = {gen: average_hamming_distance(genes) for gen, genes in self.solutions.items()}
        plt.figure(figsize=(10, 6))
        plt.plot(list(average_distances.keys()), list(average_distances.values()), color=lBlue)
        plt.xlabel('Generation')
        plt.ylabel('Average Hamming Distance')
        # plt.title('Bitstring Diversity')
        plt.grid(True)
        plt.show()

    def plot_global_best_solutions(self, show_annotations=False):
        generations = [chrom.generation for chrom in self.global_best_solutions]
        bit_strings = [chrom.bitstring for chrom in self.global_best_solutions]
        fitness_values = [chrom.fitness for chrom in self.global_best_solutions]
        idx_values = [chrom.idx for chrom in self.global_best_solutions]  # Store the idx values

        # Create the figure and axis
        _, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        # Plotting the generations vs. fitness values line
        ax.plot(generations, fitness_values, linestyle="-", color='grey', zorder=1)  # Line behind points

        # Creating a color map for the number of unique idx values
        unique_idxs = np.unique(idx_values)
        cmap = plt.cm.get_cmap('plasma', len(unique_idxs))  # Get a colormap with discrete colors

        # Plotting each point individually
        for idx in unique_idxs:
            idx_mask = [idx_val == idx for idx_val in idx_values]
            ax.scatter(np.array(generations)[idx_mask], np.array(fitness_values)[idx_mask],
                       color=cmap(idx), marker='o', label=f'chromosome: {idx}')

        # Annotating each point with its solution representation only if show_annotations is True
        if show_annotations:
            for gen, sol, fit in zip(generations, bit_strings, fitness_values):
                ax.annotate(sol, (gen, fit), textcoords="offset points", xytext=(0, 10), ha="center")

        # Adding a legend for the idx values
        # ax.legend(title="Chromosome", loc='upper left')

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Evolution of the best solution")
        ax.set_xticks(generations)
        ax.grid(True)
        ax.set_axisbelow(True)  # Ensure grid lines are behind other plot elements
        ax.set_xlim(left=0, right=self.generations)
        plt.tight_layout()
        plt.show()

    def plot_fitness_over_generations_per_quantum_chromosome(self):
        chrom_fitness = {chrom_id: [] for chrom_id in range(self.population_size)}

        # Extract fitness data
        for generation in self.best_local_per_chromosome.values():
            for chrom_id, chrom in generation.items():
                chrom_fitness[chrom_id].append(chrom.fitness)

        # Plotting
        plt.figure(figsize=(8, 6))
        for chrom_id, fitnesses in chrom_fitness.items():
            plt.plot(fitnesses, label=f'Chromosome {chrom_id}')

        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        # plt.title('Fitness of Each Chromosome Over Generations')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()
