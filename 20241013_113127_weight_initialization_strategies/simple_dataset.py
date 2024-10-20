import numpy as np


def generate_xor_data(N=100, noise=0.5):
    x = np.random.uniform(-5.0, 5.0, N) + np.random.normal(0, noise, N)
    y = np.random.uniform(-5.0, 5.0, N) + np.random.normal(0, noise, N)
    l = np.logical_xor(x > 0, y > 0).astype(int)
    return np.column_stack((x, y, l))


def generate_spiral_data(N=100, noise=0.5):
    n = N // 2

    def gen_spiral(delta_t, label):
        r = np.linspace(0, 6.0, n)
        t = 1.75 * np.linspace(0, 2 * np.pi, n) + delta_t
        x = r * np.sin(t) + np.random.uniform(-1, 1, n) * noise
        y = r * np.cos(t) + np.random.uniform(-1, 1, n) * noise
        return np.column_stack((x, y, np.full(n, label)))

    spiral1 = gen_spiral(0, 0)
    spiral2 = gen_spiral(np.pi, 1)
    return np.vstack((spiral1, spiral2))


def generate_gaussian_data(N=100, noise=0.5):
    n = N // 2

    def gen_gaussian(xc, yc, label):
        x = np.random.normal(xc, noise * 1.0 + 1.0, n)
        y = np.random.normal(yc, noise * 1.0 + 1.0, n)
        return np.column_stack((x, y, np.full(n, label)))

    gaussian1 = gen_gaussian(2, 2, 1)
    gaussian2 = gen_gaussian(-2, -2, 0)
    return np.vstack((gaussian1, gaussian2))


def generate_circle_data(N=100, noise=0.5):
    n = N // 2
    radius = 5.0

    def get_circle_label(x, y):
        return (x**2 + y**2 < (radius * 0.5) ** 2).astype(int)

    # Generate positive points inside the circle
    r_in = np.random.uniform(0, radius * 0.5, n)
    angle_in = np.random.uniform(0, 2 * np.pi, n)
    x_in = r_in * np.sin(angle_in)
    y_in = r_in * np.cos(angle_in)
    noise_in = np.random.uniform(-radius, radius, (n, 2)) * noise / 3
    points_in = np.column_stack((x_in, y_in)) + noise_in
    labels_in = get_circle_label(points_in[:, 0], points_in[:, 1])

    # Generate negative points outside the circle
    r_out = np.random.uniform(radius * 0.75, radius, n)
    angle_out = np.random.uniform(0, 2 * np.pi, n)
    x_out = r_out * np.sin(angle_out)
    y_out = r_out * np.cos(angle_out)
    noise_out = np.random.uniform(-radius, radius, (n, 2)) * noise / 3
    points_out = np.column_stack((x_out, y_out)) + noise_out
    labels_out = get_circle_label(points_out[:, 0], points_out[:, 1])

    return np.vstack(
        (
            np.column_stack((points_in, labels_in)),
            np.column_stack((points_out, labels_out)),
        )
    )


if __name__ == "__main__":
    # test data generating functions by plotting them
    import matplotlib.pyplot as plt

    def plot_data(data):
        plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap="viridis")
        plt.show()

    xor_data = generate_xor_data(N=200)
    spiral_data = generate_spiral_data(N=200)
    gaussian_data = generate_gaussian_data(N=200)
    circle_data = generate_circle_data(N=200)

    # plot_data(xor_data)
    # plot_data(spiral_data)
    # plot_data(gaussian_data)
    # plot_data(circle_data)

    gaussian_data = generate_gaussian_data(N=200)
    plot_data(gaussian_data)
    gaussian_data = generate_gaussian_data(N=200)
    plot_data(gaussian_data)
    gaussian_data = generate_gaussian_data(N=200)
    plot_data(gaussian_data)
    gaussian_data = generate_gaussian_data(N=200)
    plot_data(gaussian_data)
