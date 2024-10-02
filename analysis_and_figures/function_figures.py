import numpy as np
import matplotlib.pyplot as plt

# Define fitness functions with both local and global optima
test_functions = {
    'Sphere': {
        'function': lambda x, y: x ** 2 + y ** 2,
        'x_range': (-5.12, 5.12),
        'y_range': (-5.12, 5.12),
        'global_optima': [(0, 0)]  # Global optimum at (0, 0)
    },
    'Rastrigin': {
        'function': lambda x, y: 20 + x ** 2 + y ** 2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)),
        'x_range': (-5.12, 5.12),
        'y_range': (-5.12, 5.12),
        'global_optima': [(0, 0)]  # Global optimum at (0, 0)
    },
    'Ackley': {
        'function': lambda x, y: -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20,
        'x_range': (-5, 5),
        'y_range': (-5, 5),
        'global_optima': [(0, 0)]  # Global optimum at (0, 0)
    },
    'Schwefel': {
        'function': lambda x, y: 418.9829 * 2 - (x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y)))),
        'x_range': (-500, 500),
        'y_range': (-500, 500),
        'global_optima': [(420.9687, 420.9687)],  # Approx global optimum at (420.9687, 420.9687)
        'scaling': 5
    },
    'Rosenbrock': {
        'function': lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2,
        'x_range': (-2.048, 2.048),
        'y_range': (-2.048, 2.048),
        'global_optima': [(1, 1)]  # Global optimum at (1, 1)
    },
    'Beale': {
        'function': lambda x, y: (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2,
        'x_range': (-4.5, 4.5),
        'y_range': (-4.5, 4.5),
        'global_optima': [(3, 0.5)]  # Global optimum at (3, 0.5)
    },
    'Goldstein-Price': {
        'function': lambda x, y: (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) *
                                 (30 + (2 * x - 3 * y) ** 2 * (
                                         18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)),
        'x_range': (-2, 2),
        'y_range': (-2, 2),
        'global_optima': [(0, -1)]  # Global optimum at (0, -1)
    },
    'Booth': {
        'function': lambda x, y: (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2,
        'x_range': (-10, 10),
        'y_range': (-10, 10),
        'global_optima': [(1, 3)]  # Global optimum at (1, 3)
    },
    'Bukin N.6': {
        'function': lambda x, y: 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10),
        'x_range': (-15, -5),
        'y_range': (-3, 3),
        'global_optima': [(-10, 1)]  # Global optimum at (-10, 1)
    },
    'Matyas': {
        'function': lambda x, y: 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y,
        'x_range': (-10, 10),
        'y_range': (-10, 10),
        'global_optima': [(0, 0)]  # Global optimum at (0, 0)
    },
    'Levy': {
        'function': lambda x, y: np.sin(3 * np.pi * x) ** 2 + (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2) + (
                y - 1) ** 2 * (1 + np.sin(2 * np.pi * y) ** 2),
        'x_range': (-10, 10),
        'y_range': (-10, 10),
        'global_optima': [(1, 1)]  # Global optimum at (1, 1)
    },
    'Three-hump Camel': {
        'function': lambda x, y: 2 * x ** 2 - 1.05 * x ** 4 + (x ** 6) / 6 + x * y + y ** 2,
        'x_range': (-5, 5),
        'y_range': (-5, 5),
        'global_optima': [(0, 0)],
        'scaling': 50
    },
    'Easom': {
        'function': lambda x, y: -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2)),
        'x_range': (-100, 100),
        'y_range': (-100, 100),
        'global_optima': [(np.pi, np.pi)],
        'scaling': 100
    },
    'Cross-in-Tray': {
        'function': lambda x, y: -0.0001 * (
                np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - np.sqrt(x ** 2 + y ** 2) / np.pi))) + 1) ** 0.1,
        'x_range': (-10, 10),
        'y_range': (-10, 10),
        'global_optima': [(1.3491, -1.3491), (-1.3491, 1.3491), (1.3491, 1.3491), (-1.3491, -1.3491)]
    },
    'Eggholder': {
        'function': lambda x, y: -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(
            np.sqrt(np.abs(x - (y + 47)))),
        'x_range': (-512, 512),
        'y_range': (-512, 512),
        'global_optima': [(512, 404.2319)]  # Global optimum at (512, 404.2319)
    },
    'Holder Table': {
        'function': lambda x, y: -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x ** 2 + y ** 2) / np.pi))),
        'x_range': (-10, 10),
        'y_range': (-10, 10),
        'global_optima': [(8.05502, 9.66459), (-8.05502, 9.66459), (8.05502, -9.66459), (-8.05502, -9.66459)]
    },
    'McCormick': {
        'function': lambda x, y: np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1,
        'x_range': (-1.5, 4),
        'y_range': (-3, 4),
        'global_optima': [(-0.54719, -1.54719)],
        'scaling': 20
    },
    'Schaffer N.2': {
        'function': lambda x, y: 0.5 + (np.sin(x ** 2 - y ** 2) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2,
        'x_range': (-100, 100),
        'y_range': (-100, 100),
        'global_optima': [(0, 0)],
        'scaling': 100
    }
}

fitness = {
    'Schwefel': {
        'function': lambda x, y: 418.9829 * 2 - (x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y)))),
        'x_range': (-500, 500),
        'y_range': (-500, 500),
        'global_optima': [(420.9687, 420.9687)],  # Approx global optimum at (420.9687, 420.9687)
        'scaling': 10
    },
    'Holder Table': {
        'function': lambda x, y: -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x ** 2 + y ** 2) / np.pi))),
        'x_range': (-10, 10),
        'y_range': (-10, 10),
        'global_optima': [(8.05502, 9.66459), (-8.05502, 9.66459), (8.05502, -9.66459), (-8.05502, -9.66459)],
        'scaling': 50
    },
}

figsize = (12, 8)
dpi = 120
azim = 30
default_scaling = 10

for name, details in test_functions.items():
    func = details['function']
    x_min, x_max = details['x_range']
    y_min, y_max = details['y_range']
    global_optima = details['global_optima']
    scaling = details.get('scaling', default_scaling)

    # Generate grid for x and y values
    num_points_x = int((x_max - x_min) * scaling)
    num_points_y = int((y_max - y_min) * scaling)

    x = np.linspace(x_min, x_max, num_points_x)
    y = np.linspace(y_min, y_max, num_points_y)
    x, y = np.meshgrid(x, y)
    z = func(x, y)

    # Plotting the surface
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_proj_type('ortho')
    ax.set_box_aspect([1, 1, .7])
    ax.view_init(azim=azim)

    # Handle global optima
    global_z_values = [func(global_x, global_y) for global_x, global_y in global_optima]
    for (global_x, global_y), global_z in zip(global_optima, global_z_values):
        ax.scatter(global_x, global_y, global_z, color='r', s=100)

    # Labels and title
    ax.set_xlabel('Solution Space (x)')
    ax.set_ylabel('Solution Space (y)')
    ax.set_zlabel('Fitness Function f(x, y)')

    # Use tight_layout with padding
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
