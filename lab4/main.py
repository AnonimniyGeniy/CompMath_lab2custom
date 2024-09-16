import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define various approximation functions as per the assignment requirements

def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def exponential(x, a, b):
    return a * np.exp(b * x)

def logarithmic(x, a, b):
    return a * np.log(x) + b

def power(x, a, b):
    return a * x**b

# Calculate the coefficient of determination (R-squared)
def calculate_r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)

# Calculate Pearson correlation coefficient for linear function
def pearson_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

def create_table(func, interval, num_points=11):
    """Create a table of values for the given function over the specified interval."""
    x = np.linspace(interval[0], interval[1], num_points)
    y = func(x)
    return x, y

def table_research(func, interval):
    """Perform the table part of the research."""
    x, y = create_table(func, interval)
    
    print("\nTable of values:")
    for xi, yi in zip(x, y):
        print(f"x: {xi:.2f}, y: {yi:.4f}")
    
    # Linear approximation
    popt_linear, _ = curve_fit(linear, x, y)
    y_linear = linear(x, *popt_linear)
    rmse_linear = np.sqrt(np.mean((y - y_linear)**2))
    
    # Quadratic approximation
    popt_quad, _ = curve_fit(quadratic, x, y)
    y_quad = quadratic(x, *popt_quad)
    rmse_quad = np.sqrt(np.mean((y - y_quad)**2))
    
    print("\nLinear approximation:")
    print(f"Coefficients: {popt_linear}")
    print(f"RMSE: {rmse_linear:.3f}")
    
    print("\nQuadratic approximation:")
    print(f"Coefficients: {popt_quad}")
    print(f"RMSE: {rmse_quad:.3f}")
    
    # Determine the best approximation
    if rmse_linear < rmse_quad:
        print("\nLinear approximation is better")
    else:
        print("\nQuadratic approximation is better")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='red', label='Original data')
    plt.plot(x, y_linear, label='Linear approximation')
    plt.plot(x, y_quad, label='Quadratic approximation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Function Approximation (Table Research)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to implement the least squares method for function approximation.
    This program fulfills the requirements of Laboratory Work No. 4:
    "Function Approximation by Least Squares Method"
    """

    # Table part of the research
    # Function for variant 5: y = 6x / (x^4 + 5)
    def variant_function(x):
        return 6 * x / (x**4 + 5)
    
    interval = (0, 2)
    
    print("Table part of the research:")
    table_research(variant_function, interval)
    
    # Input data (requirement: accept 8 to 12 points)
    try:
        x = np.array([float(i) for i in input("Enter x values separated by space: ").split()])
        y = np.array([float(i) for i in input("Enter y values separated by space: ").split()])
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    if len(x) != len(y) or len(x) < 8 or len(x) > 12:
        print("Invalid input. Please provide 8 to 12 points.")
        return

    # Define the functions to be investigated as per the assignment
    functions = [
        ("Linear", linear),
        ("Quadratic", quadratic),
        ("Cubic", cubic),
        ("Exponential", exponential),
        ("Logarithmic", logarithmic),
        ("Power", power)
    ]

    best_function = None
    best_rmse = float('inf')
    results = []

    # Prepare the plot
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, color='red', label='Original data')

    # Investigate all specified functions
    for name, func in functions:
        try:
            # Fit the function and calculate necessary metrics
            popt, _ = curve_fit(func, x, y)
            y_pred = func(x, *popt)
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            r_squared = calculate_r_squared(y, y_pred)
            
            results.append((name, popt, rmse, r_squared))
            
            # Determine the best approximating function
            if rmse < best_rmse:
                best_rmse = rmse
                best_function = name

            # Plot the approximating function
            plt.plot(x, y_pred, label=f'{name} (RMSE: {rmse:.3f})')
            
            # Output results
            print(f"\n{name} function:")
            print(f"Coefficients: {popt}")
            print(f"RMSE: {rmse:.3f}")
            print(f"R-squared: {r_squared:.3f}")
            
            # Calculate Pearson correlation coefficient for linear function
            if name == "Linear":
                correlation = pearson_correlation(x, y)
                print(f"Pearson correlation coefficient: {correlation:.3f}")

        except:
            print(f"Could not fit {name} function")

    # Output the best approximating function
    print(f"\nBest approximating function: {best_function}")

    # Interpret the coefficient of determination
    r_squared = next(r for name, _, _, r in results if name == best_function)
    if r_squared > 0.7:
        print("The model explains the data well")
    elif r_squared > 0.5:
        print("The model moderately explains the data")
    else:
        print("The model poorly explains the data")

    # Finalize and display the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Function Approximation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
