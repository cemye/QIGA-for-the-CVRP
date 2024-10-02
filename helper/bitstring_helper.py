import math

import numpy as np
from numpy import ndarray
from qiskit.result import Result
from scipy.spatial.distance import pdist


def bitstring_to_array(bitstring: str) -> np.ndarray:
    return np.astype(np.array(list(bitstring), dtype=int), bool)


def array_to_bitstring(array: np.ndarray) -> str:
    return ''.join(np.astype(array, str))


# --------------------
# DECODING FUNCTIONS
# --------------------

def binary_to_gray(array):
    """Convert a Boolean binary array to a Boolean Gray code array."""
    binary = array  # Ensure integer for bitwise operations
    gray = np.zeros_like(binary)
    gray[0] = binary[0]
    gray[1:] = binary[:-1] ^ binary[1:]
    return gray


def gray_to_binary(gray_array):
    """Convert a Boolean Gray code array to a Boolean binary array."""
    binary = np.zeros_like(gray_array)
    binary[0] = gray_array[0]
    for i in range(1, len(gray_array)):
        binary[i] = binary[i - 1] ^ gray_array[i]
    return binary


def bool_array_to_single_int(bool_array):
    """Convert a Boolean array to a single integer."""
    return int(bool_array.dot(1 << np.arange(bool_array.size - 1, -1, -1, dtype=object)))


def single_int_to_bool_array(value, length):
    """Convert an integer to a Boolean array of a specified length."""
    return np.array([(value >> i) & 1 for i in range(length)][::-1], dtype=int)


def bool_array_to_segmented_ints(bool_array, segment_sizes) -> ndarray:
    """Convert a Boolean array into segmented integers based on segment sizes."""
    if isinstance(segment_sizes, int):
        # Assuming perfect divisibility by segment_sizes, prepare repeat list
        segment_sizes = [segment_sizes] * (len(bool_array) // segment_sizes)

    indices = np.cumsum(segment_sizes)  # Calculate ending indices for each segment
    segmented_ints = []
    start = 0

    for end in indices:
        segment = bool_array[start:end]
        # Handle edge cases by ensuring segment size is a multiple of 8
        packed_bits = np.packbits(np.astype(segment, np.uint8), bitorder='big')
        segment_int = int.from_bytes(packed_bits, byteorder='big')
        # Adjust for non-multiples of 8
        segment_int >>= 8 * (len(packed_bits)) - len(segment)
        segmented_ints.append(segment_int)
        start = end

    return np.array(segmented_ints, dtype=object)


def segmented_ints_to_bool_array(segmented_ints, segment_sizes):
    """Convert an array of segmented integers back to a Boolean array."""
    if isinstance(segment_sizes, int):
        segment_sizes = [segment_sizes] * len(segmented_ints)

    # Preallocate the result array
    total_bits = sum(segment_sizes)
    bool_array = np.zeros(total_bits, dtype=bool)

    start = 0
    for value, size in zip(segmented_ints, segment_sizes):
        # Convert the integer to a bit array using bit manipulation
        # Right-shift and mask to extract each bit in the segment
        for i in range(size):
            bool_array[start + size - 1 - i] = (value >> i) & 1
        start += size

    return bool_array


# --------------------
# CALCULATING BITS NEEDED FUNCTIONS
# --------------------

def bits_needed_for_integer(integer: int | np.ndarray) -> int | ndarray:
    return np.ceil(np.log2(np.array(integer) + 1)).astype(int)


def bits_needed_for_permutation_as_int(problem_size: int) -> ndarray:
    """Calculate the number of bits needed to represent a factoradic number of a given problem size."""
    return np.array([int(math.log2(math.factorial(problem_size) - 1)) + 1])


def segmented_sizes_for_factoradic(problem_size: int) -> ndarray:
    """Calculate the number of bits for each segment needed to represent a factoradic number of a given problem size
    excluding the depot with different segment sizes."""
    n_values = np.arange(problem_size, 0, -1)
    return bits_needed_for_integer(n_values)


def segmented_sizes_for_vehicle_assignment(problem_size: int, vehicles: int) -> ndarray:
    """Calculate the number of bits for each segment needed to represent a vehicle assignment problem."""
    return np.array([bits_needed_for_integer(vehicles)] * problem_size)


# --------------------
# SPLITTING FUNCTIONS
# --------------------

def split_results_into_chromosome_chunks_with_shots(population_size: int, qubit_count: int, results: Result):
    counts = results.get_counts()
    bitstrings = list(counts.keys())
    # Convert all bitstrings to a single large array of bools
    all_bits = np.fromiter(''.join(bitstrings), dtype='i1')
    all_bits = all_bits.reshape(len(counts.keys()), qubit_count)

    # Split the bit array into chromosomes
    result = np.split(all_bits, population_size, axis=1)  # matrix
    return [list(x) for x in result]


# --------------------
# MATH FUNCTIONS
# --------------------

def hamming_distance(gene1, gene2):
    """Calculate the Hamming distance between two genes' bitstrings, optimized for boolean numpy arrays."""
    # Assuming gene1.bitstring and gene2.bitstring are numpy arrays of dtype=bool
    return np.sum(gene1.bitstring != gene2.bitstring)


def average_hamming_distance(genes):
    """Calculate the average Hamming distance among a list of ClassicalChromosome objects."""
    if not genes:
        return 0
    gene_array = np.array([gene.bitstring for gene in genes])
    hamming_distances = pdist(gene_array, 'hamming')
    hamming_distances *= gene_array.shape[1]  # Scale by number of bits to get bit counts
    return np.mean(hamming_distances)


# --------------------
# HELPER FUNCTIONS
# --------------------

def bitstring_to_single_int(bitstring: ndarray, use_gray: bool) -> int:
    """Converts a boolean numpy array bitstring into a single integer using specified encoding."""
    bitstring = gray_to_binary(bitstring) if use_gray else bitstring
    return bool_array_to_single_int(bitstring)


def bitstring_to_int_list(bitstring: ndarray, segment_sizes: ndarray, use_gray: bool,
                          add_one: bool = False) -> ndarray:
    """Converts a boolean numpy array bitstring into a numpy array of integers using specified encoding."""
    bitstring = gray_to_binary(bitstring) if use_gray else bitstring
    int_list = bool_array_to_segmented_ints(bitstring, segment_sizes)
    int_list += 1 if add_one else 0
    return int_list.astype(object)


def single_int_to_factoradic(n: int, size: int) -> np.ndarray:
    """Converts a single integer to factoradic representation including the zero digit at the end"""
    factoradic = np.zeros(size, dtype=int)
    for i in range(1, size + 1):
        factoradic[-i] = n % i
        n //= i
    return factoradic


def permutation_to_single_int(permutation: np.ndarray) -> int:
    total_nodes = len(permutation)
    available_nodes = np.arange(1, total_nodes + 1)
    index, factor = 0, 1

    for i in range(total_nodes):
        node = permutation[i]
        node_index = np.where(available_nodes == node)[0][0]
        index += node_index * factor
        available_nodes = np.delete(available_nodes, node_index)
        factor *= total_nodes - i
    return index


def factoradic_to_single_int(factoradic: np.ndarray) -> int:
    """Converts a factoradic number (array) to a single integer."""
    if not isinstance(factoradic, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")

    n, factorial = 0, 1
    # Ensure calculations use Python's arbitrary-precision integers
    for i, digit in enumerate(reversed(np.astype(factoradic, int))):
        n += int(digit) * factorial  # Convert to Python int for safe large number handling
        factorial *= (i + 1)

    return n


def repair_factoradic_with_modulo(factoradic_array: ndarray) -> ndarray:
    max_values = np.arange(factoradic_array.size + 1, 1, -1)  # calculate the modulo values for each digit
    return np.mod(factoradic_array, max_values)


def repair_factoradic_with_scaling(factoradic: ndarray, segment_sizes: ndarray,
                                   max_segment_sizes: ndarray) -> ndarray:
    factoradic = np.array(factoradic, dtype=int)
    """Repairs a factoradic array by scaling each digit based on segment sizes efficiently."""
    if len(factoradic) == len(segment_sizes) + 1 and len(factoradic) == len(max_segment_sizes) + 1:
        segment_sizes = segment_sizes[:-1]
        max_segment_sizes = max_segment_sizes[:-1]
    elif len(factoradic) != len(segment_sizes) or len(factoradic) != len(max_segment_sizes):
        raise ValueError("Length of factoradic array, segment_sizes, and max_segment_sizes must be compatible.")

    max_binary_values = (1 << np.array(segment_sizes)) - 1
    scaled_values = factoradic / max_binary_values * max_segment_sizes
    return np.round(scaled_values).astype(int)


# --------------------
# PERMUTATION CONVERSION FUNCTIONS
# --------------------


def factoradic_to_permutation(factoradic: np.ndarray, append_zero: bool = True) -> np.ndarray:
    """Convert factoradic number to permutation using numpy arrays."""

    factoradic = np.append(factoradic, 0) if append_zero else factoradic
    n = len(factoradic)
    items = np.arange(n, dtype=object)
    permutation = np.empty(n, dtype=object)
    # Adjusting factoradic values during iteration
    for i in range(n):
        factor = factoradic[i]
        if factor >= len(items):
            raise ValueError("Factor out of bounds. Please check the factoradic numbers.")
        permutation[i] = items[factor]
        items = np.delete(items, factor)  # Remove used item

    return permutation


def permutation_to_factoradic(permutation: np.ndarray) -> np.ndarray:
    """Convert permutation to factoradic number."""
    n = len(permutation)
    factoradic = np.empty(n, dtype=int)
    items = np.arange(n)

    for i in range(n):
        index = np.where(items == permutation[i])[0][0]
        factoradic[i] = index
        items = np.delete(items, index)  # Remove used item

    return factoradic[:-1]  # Remove the last element, which is always 0
