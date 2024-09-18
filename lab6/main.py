import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

class ODESolver:
    def __init__(self, f: Callable[[float, float], float], y0: float, t0: float, t_end: float, h: float, epsilon: float):
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.t_end = t_end
        self.h = h
        self.epsilon = epsilon

    def improved_euler(self) -> Tuple[List[float], List[float]]:
        t = np.arange(self.t0, self.t_end + self.h, self.h)
        y = np.zeros(len(t))
        y[0] = self.y0

        for i in range(1, len(t)):
            k1 = self.f(t[i-1], y[i-1])
            k2 = self.f(t[i], y[i-1] + self.h * k1)
            y[i] = y[i-1] + 0.5 * self.h * (k1 + k2)

        return t.tolist(), y.tolist()

    def runge_kutta_4(self) -> Tuple[List[float], List[float]]:
        t = np.arange(self.t0, self.t_end + self.h, self.h)
        y = np.zeros(len(t))
        y[0] = self.y0

        for i in range(1, len(t)):
            k1 = self.f(t[i-1], y[i-1])
            k2 = self.f(t[i-1] + 0.5*self.h, y[i-1] + 0.5*self.h*k1)
            k3 = self.f(t[i-1] + 0.5*self.h, y[i-1] + 0.5*self.h*k2)
            k4 = self.f(t[i-1] + self.h, y[i-1] + self.h*k3)
            y[i] = y[i-1] + (self.h/6) * (k1 + 2*k2 + 2*k3 + k4)

        return t.tolist(), y.tolist()

    def milne(self) -> Tuple[List[float], List[float]]:
        t = np.arange(self.t0, self.t_end + self.h, self.h)
        y = np.zeros(len(t))
        y[0] = self.y0

        # Use Runge-Kutta 4 to get the first 4 points
        for i in range(1, 4):
            k1 = self.f(t[i-1], y[i-1])
            k2 = self.f(t[i-1] + 0.5*self.h, y[i-1] + 0.5*self.h*k1)
            k3 = self.f(t[i-1] + 0.5*self.h, y[i-1] + 0.5*self.h*k2)
            k4 = self.f(t[i-1] + self.h, y[i-1] + self.h*k3)
            y[i] = y[i-1] + (self.h/6) * (k1 + 2*k2 + 2*k3 + k4)

        for i in range(4, len(t)):
            # Predictor
            y_pred = y[i-4] + (4*self.h/3) * (2*self.f(t[i-3], y[i-3]) - self.f(t[i-2], y[i-2]) + 2*self.f(t[i-1], y[i-1]))
            
            # Corrector
            y[i] = y[i-2] + (self.h/3) * (self.f(t[i-2], y[i-2]) + 4*self.f(t[i-1], y[i-1]) + self.f(t[i], y_pred))

        return t.tolist(), y.tolist()

def main():
    equations = [
        ("y' = y", lambda t, y: y),
        ("y' = t*y", lambda t, y: t*y),
        ("y' = y^2", lambda t, y: y**2),
    ]

    print("Choose an equation:")
    for i, (eq, _) in enumerate(equations):
        print(f"{i+1}. {eq}")

    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(equations):
                f = equations[choice-1][1]
                break
            else:
                print(f"Please enter a number between 1 and {len(equations)}.")
        except ValueError:
            print("Please enter a valid integer.")

    def get_float_input(prompt):
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Please enter a valid number.")

    y0 = get_float_input("Enter initial y value (y0): ")
    t0 = get_float_input("Enter initial t value (t0): ")
    t_end = get_float_input("Enter end t value (t_end): ")
    
    while True:
        h = get_float_input("Enter step size (h): ")
        if h > 0:
            break
        print("Step size must be positive.")

    while True:
        epsilon = get_float_input("Enter desired accuracy (epsilon): ")
        if epsilon > 0:
            break
        print("Accuracy must be positive.")

    solver = ODESolver(f, y0, t0, t_end, h, epsilon)

    methods = [
        ("Improved Euler", solver.improved_euler),
        ("Runge-Kutta 4", solver.runge_kutta_4),
        ("Milne", solver.milne),
    ]

    plt.figure(figsize=(12, 8))
    for name, method in methods:
        t, y = method()
        plt.plot(t, y, label=name)
        print(f"\n{name} method:")
        print("t\t\ty")
        for ti, yi in zip(t, y):
            print(f"{ti:.6f}\t{yi:.6f}")

    # Plot exact solution if available
    if choice == 1:
        t = np.linspace(t0, t_end, 1000)
        y_exact = y0 * np.exp(t - t0)
        plt.plot(t, y_exact, 'k--', label='Exact solution')

    plt.xlabel('t (time)')
    plt.ylabel('y (solution)')
    plt.title('Numerical Solutions of ODE')
    plt.legend()
    plt.grid(True)

    # Add text explaining t and y in the bottom left corner
    plt.text(0.02, 0.02, 't: time\ny: solution', transform=plt.gca().transAxes,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.show()

if __name__ == "__main__":
    main()