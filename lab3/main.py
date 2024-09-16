import math

# Функция f(x, equation) определяет различные подынтегральные функции

def f(x, equation):
    if equation == 1:
        return x**2
    elif equation == 2:
        return -x**3 - x**2 - 2*x + 1
    elif equation == 3:
        return -6*x**3 + x**2 - x + 5
    elif equation == 4:
        return math.cos(x)
    else:
        raise ValueError("Неизвестное уравнение.")

# Метод прямоугольников (левые, правые, средние)
def rectangle(a, b, e, equation, mode, p):
    """
    a, b - пределы интегрирования
    e - заданная точность
    Использует правило Рунге для оценки погрешности
    """
    n = 2
    h = (b - a) / n
    prev_sum = 0
    
    while True:
        sum = 0
        for i in range(n):
            if mode == 1:  # левые прямоугольники
                x = a + i * h
            elif mode == 2:  # правые прямоугольники
                x = a + (i + 1) * h
            else:  # средние прямоугольники
                x = a + (i + 0.5) * h
            sum += f(x, equation)
        
        sum *= h
        if abs(sum - prev_sum) < e:
            return sum, n
        
        prev_sum = sum
        n *= 2
        h /= 2

# Метод трапеций
def trapezoid(a, b, n, equation):
    """
    Формула метода трапеций:
    ∫[a,b] f(x)dx ≈ (b-a)/n * [(f(a) + f(b))/2 + sum(f(x_i))]
    где x_i = a + i*(b-a)/n, i = 1, 2, ..., n-1
    """
    h = (b - a) / n
    sum = 0.5 * (f(a, equation) + f(b, equation))
    for i in range(1, n):
        x = a + i * h
        sum += f(x, equation)
    return h * sum

# Метод Рунге с использованием метода трапеций
def runge(a, b, e, equation, p):
    """
    Правило Рунге для оценки погрешности:
    |I - I_n| ≈ |I_n - I_{n/2}| / (2^p - 1)
    где p - порядок точности метода (для трапеций p = 2)
    """
    n = 2
    prev_integral = trapezoid(a, b, n, equation)
    
    while True:
        n *= 2
        current_integral = trapezoid(a, b, n, equation)
        if abs(current_integral - prev_integral) / (2**p - 1) < e:
            return current_integral, n
        prev_integral = current_integral

# Метод Симпсона
def simpson(a, b, n, equation):
    """
    Формула метода Симпсона:
    ∫[a,b] f(x)dx ≈ (b-a)/(3n) * [f(a) + f(b) + 4*sum(f(x_i)) + 2*sum(f(x_j))]
    где x_i = a + (2i-1)*(b-a)/(2n), i = 1, 2, ..., n
    и x_j = a + 2j*(b-a)/(2n), j = 1, 2, ..., n-1
    """
    h = (b - a) / n
    sum = f(a, equation) + f(b, equation)
    
    for i in range(1, n, 2):
        x = a + i * h
        sum += 4 * f(x, equation)
    
    for i in range(2, n-1, 2):
        x = a + i * h
        sum += 2 * f(x, equation)
    
    return (h / 3) * sum

# Метод Рунге с использованием метода Симпсона
# Реализация пункта 4 обязательного задания
def runge_simpson(a, b, e, equation, p):
    """
    Применение правила Рунге к методу Симпсона
    p = 4 для метода Симпсона
    """
    n = 2
    prev_integral = simpson(a, b, n, equation)
    
    while True:
        n *= 2
        current_integral = simpson(a, b, n, equation)
        if abs(current_integral - prev_integral) / (2**p - 1) < e:
            return current_integral, n
        prev_integral = current_integral

# Основная программа
equation = int(input("Выберите функцию:\n1. x^2\n2. -x^3 - x^2 - 2x + 1\n3. -6x^3 + x^2 - x + 5\n4. cos(x)\n> "))
lower = int(input("Введите нижний предел интегрирования: "))
upper = int(input("Введите верхний предел интегрирования: "))

method = int(input("Выберите метод решения интеграла:\n1. Метод прямоугольников\n2. Метод трапеций.\n3. Метод Симпсона.\n> "))
if method == 1:
    e = float(input("Введите точность: "))
    res_left = rectangle(lower, upper, e, equation, 1, 4)
    print("Метод левых прямоугольников:")
    print(f"Значение интеграла: I  = {res_left[0]}; "
          f"Число разбиения интервала интегрирования для достижения требуемой точности: n = {res_left[1]}")

    # Метод правых прямоугольников
    res_right = rectangle(lower, upper, e, equation, 2, 4)
    print("Метод правых прямоугольников:")
    print(f"Значение интеграла: I  = {res_right[0]}; "
          f"Число разбиения интервала интегрирования для достижения требуемой точности: n = {res_right[1]}")

    # Метод средних прямоугольников
    res_avg = rectangle(lower, upper, e, equation, 3, 4)
    print("Метод средних прямоугольников:")
    print(f"Значение интеграла: I  = {res_avg[0]}; "
          f"Число разбиения интервала интегрирования для достижения требуемой точности: n = {res_avg[1]}")
elif method == 2:
    e = float(input("Введите точность: "))
    res = runge(lower, upper, e, equation, 4)
    print(f"Значение интеграла: I  = {res[0]}; "
          f"Число разбиения интервала интегрирования для достижения требуемой точности: n = {res[1]}")
elif method == 3:
    e = float(input("Введите точность: "))
    res = runge_simpson(lower, upper, e, equation, 4)
    print(f"Значение интеграла: I  = {res[0]};"
          f"Число разбиения интервала интегрирования для достижения требуемой точности: n = {res[1]}")
else:
    raise ValueError("Неизвестный режим.")
