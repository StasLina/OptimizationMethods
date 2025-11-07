import numpy as np

def function(x):
    A = np.array([[2.0, 0.5],
                  [0.5, 1.0]])
    return x @ A @ x

def grad_f(x):
    A = np.array([[2.0, 0.5],
                  [0.5, 1.0]])
    return 2 * A @ x   # ∇f(x) = 2 A x

def optimal_step(x, A):
    """
    Аналитический оптимальный шаг t_k для f(x) = x^T A x
    """
    g = grad_f(x)  # = 2 A x
    numerator = g.T @ g          # ||g||^2
    denominator = 2 * (g.T @ A @ g)
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator

def gradient_descent_optimal_step(
    x0,
    tol=1e-6,
    max_iter=1000,
    verbose=True
):
    """
    Метод наискорейшего градиентного спуска (с аналитическим шагом)
    """
    x = np.array(x0, dtype=float)
    A = np.array([[2.0, 0.5],
                  [0.5, 1.0]])
    
    x_hist = [x.copy()]
    f_hist = [function(x)]
    
    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        
        if verbose and k % 5 == 0:
            print(f"Iter {k:4d} | x = [{x[0]:.4f}, {x[1]:.4f}] | f(x) = {f_hist[-1]:.6e} | ||∇f|| = {grad_norm:.2e}")
        
        if grad_norm < tol:
            if verbose:
                print(f"✅ Сходимость достигнута на итерации {k}. ||∇f|| = {grad_norm:.2e} < {tol}")
            break
        
        # Вычисляем оптимальный шаг t_k
        t_k = optimal_step(x, A)
        
        # Делаем шаг
        x = x - t_k * grad
        x_hist.append(x.copy())
        f_hist.append(function(x))
    else:
        if verbose:
            print("⚠️ Достигнуто максимальное число итераций. Сходимость не достигнута.")
    
    return np.array(x_hist), np.array(f_hist)

# Пример запуска
if __name__ == "__main__":
    x0 = np.array([2.0, -1.0])
    x_hist, f_hist = gradient_descent_optimal_step(x0, tol=1e-8, max_iter=50)

    x_opt = x_hist[-1]
    print("\n🏁 Результат:")
    print(f"Найденный минимум: x* = {x_opt}")
    print(f"f(x*) = {function(x_opt):.2e}")
    print("Точный минимум: x* = [0, 0]")