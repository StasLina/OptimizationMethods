def linear_combination(a, x):
    if len(a) != len(x):
        raise ValueError("Векторы должны быть одинаковой длины")
    return sum(ai * xi for ai, xi in zip(a, x))

def quadratic_form(A, x):
    """Вычисляет f(x) = x^T A x"""
    n = len(x)
    if len(A) != n or any(len(row) != n for row in A):
        raise ValueError("Несовместимые размеры матрицы и вектора")
    Ax = [linear_combination(A[i], x) for i in range(n)]
    return linear_combination(x, Ax)

class HookeJeevesMethod:
    def __init__(self, matrix_A):
        self.A = matrix_A
        self.n = len(matrix_A)

    def f(self, x):
        """Целевая функция: f(x) = x^T A x"""
        return quadratic_form(self.A, x)

    def exploratory_search(self, x_base, delta):
        """
        Исследующий поиск:
        Поочерёдно проверяет +delta и -delta по каждой координате.
        Обновляет точку сразу при улучшении.
        """
        x = x_base[:]  # копия — будем модифицировать

        for i in range(self.n):
            f_current = self.f(x)

            # Пробуем шаг +delta по i-й координате
            x[i] += delta
            f_plus = self.f(x)
            if f_plus < f_current:
                continue  # улучшение найдено — оставляем новое значение

            # Шаг +delta не помог → возвращаем и пробуем -delta
            x[i] -= 2 * delta  # теперь x[i] = x_base[i] - delta
            f_minus = self.f(x)
            if f_minus < f_current:
                continue  # улучшение найдено — оставляем

            # Ни +delta, ни -delta не помогли → возвращаем исходное значение
            x[i] += delta

        return x

    def method_huka_jibsa(self, x0, eps=1e-6, delta=1.0, alpha=2.0, max_iter=1000):
        """
        Метод Хука-Дживса.

        Параметры:
        - x0: начальная точка (список)
        - eps: точность (минимальный допустимый шаг)
        - delta: начальный шаг
        - alpha: коэффициент уменьшения шага (обычно 2)
        - max_iter: максимальное число итераций

        Возвращает:
        - x_opt: найденная точка минимума
        """
        if eps <= 0:
            raise ValueError("eps должно быть > 0")
        if delta <= 0:
            raise ValueError("delta должно быть > 0")
        if alpha <= 1:
            raise ValueError("alpha должно быть > 1")

        x_current = x0[:]
        iteration = 0

        while delta >= eps and iteration < max_iter:
            # 1. Исследующий поиск из текущей точки
            x_explored = self.exploratory_search(x_current, delta)

            if self.f(x_explored) < self.f(x_current):
                # 2. Успешный поиск → делаем паттерн-поиск
                # Направление: x_explored - x_current
                direction = [x_explored[i] - x_current[i] for i in range(self.n)]
                # Паттерн-точка: x_explored + direction
                x_pattern = [x_explored[i] + direction[i] for i in range(self.n)]
                # Исследуем окрестность паттерн-точки
                x_new = self.exploratory_search(x_pattern, delta)

                # Выбираем лучшую из x_explored и x_new
                x_current = x_new if self.f(x_new) < self.f(x_explored) else x_explored
            else:
                # 3. Нет улучшения → уменьшаем шаг
                delta /= alpha

            iteration += 1

        return x_current


# ==============================
# Пример использования
# ==============================
if __name__ == "__main__":
    # Пример: f(x) = 2*x1^2 + x1*x2 + x2^2
    # Матрица A (должна быть симметричной!)
    A = [
        [2.0, 0.5],
        [0.5, 1.0]
    ]

    x0 = [3.0, 5.0]
    eps = 1e-6

    optimizer = HookeJeevesMethod(A)
    x_opt = optimizer.method_huka_jibsa(x0, eps=eps, delta=1.0)

    f_initial = optimizer.f(x0)
    f_opt = optimizer.f(x_opt)


    print(f"Начальная точка: {x0}")
    print(f"Начальное значение f: {f_initial:.6f}")
    print(f"Найденный минимум: [{x_opt[0]:.6f}, {x_opt[1]:.6f}]")
    print(f"Значение f в минимуме: {f_opt:.2e}")

    # Теоретический минимум для положительно определённой квадратичной формы — в нуле.
    # При правильной работе алгоритм должен приблизиться к [0, 0].

    # единичная матрица
    # базис вектор
    # 