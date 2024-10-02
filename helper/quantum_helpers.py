from typing import List

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.result import Result
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


def simulate(circuit: QuantumCircuit, simulator: AerSimulator, shots: int = 1) -> Result:
    """
        Simulate a quantum circuit using the specified method and number of shots.

        Parameters:
            circuit (QuantumCircuit): The quantum circuit to simulate.
            simulator: The simulator object to use for simulation.
            shots (int): The number of simulation shots.

        Returns:
            Result: The result object containing the simulation outcomes.
        """
    return simulator.run(circuit.reverse_bits(), shots=shots).result()


def simulate_and_plot(circuit: QuantumCircuit, shots: int = 1000):
    """
    Simulate a quantum circuit and plot the histogram of outcomes.

    Parameters:
        circuit (QuantumCircuit): The quantum circuit to simulate.
        shots (int): The number of simulation shots.

    Returns:
        Figure: A matplotlib figure showing the histogram of the simulation results.
    """
    result = simulate(circuit, AerSimulator(method="matrix_product_state"), shots)
    counts = result.get_counts(circuit)
    return plot_histogram(counts)


def generate_q_registers(population_size: int, chromosome_size: int) -> List[QuantumRegister]:
    """
    Generate a list of quantum registers for a population.

    Parameters:
        population_size (int): The number of individuals in the population.
        chromosome_size (int): The number of qubits in each chromosome.

    Returns:
        List[QuantumRegister]: A list of quantum registers for each individual.
    """
    return [QuantumRegister(chromosome_size, name=("G" + str(k))) for k in range(population_size)]


def draw(circuit: QuantumCircuit):
    return circuit.draw("mpl", style="clifford", fold=-1, scale=0.8)
