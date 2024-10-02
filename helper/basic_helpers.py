import numpy as np

TWO_PI = 2 * np.pi
PI_HALF = np.pi / 2


def convert_global_to_idxq(chromosome_size: int, global_idq: int) -> tuple[int, int]:
    idx = global_idq // chromosome_size
    idq = global_idq % chromosome_size
    return idx, idq


def convert_idq_to_global(chromosome_size: int, idx: int, idq: int) -> int:
    return chromosome_size * idx + idq


def normalize_angles(angles: np.ndarray) -> np.ndarray:
    """Normalize angles to [-π/2, π/2] with mirroring at π/2 and -π/2 via vectorized operations."""
    angles = angles % TWO_PI  # Normalize angle to [0, 2π] for all elements
    angles = angles - TWO_PI * (angles > np.pi)  # Adjust to [-π, π] for all elements

    # Now, apply the condition to each element using vectorized operations
    angles = np.where(angles > PI_HALF, np.pi - angles, angles)
    angles = np.where(angles < -PI_HALF, -np.pi - angles, angles)
    return angles


def euclidean_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Calculate the Euclidean distances between arrays of coordinates."""
    return np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))


# --------------------
# FORMATTING FUNCTIONS
# --------------------

def leading_zero_formatter(to_format: int, length: int) -> str:
    return str(to_format).zfill(length)


def get_str_len(number: int) -> int:
    return len(str(number))


def convert_to_printable_time(duration: int | float) -> str:
    duration = round(duration)
    return f'{duration // 60}m {duration % 60}s' if duration >= 120 else f'{duration}s'

