import numpy as np

def function(x):
    A = np.array([[2.0, 0.5],
                  [0.5, 1.0]])
    return x @ A @ x

def grad_f(x):
    A = np.array([[2.0, 0.5],
                  [0.5, 1.0]])
    return 2 * A @ x   # ∇f(x) = 2 A x

def gradient_descent(
    x0,
    lr=0.1,
    tol=1e-6,
    max_iter=1000,
    verbose=True
):
    """
    Метод градиентного спуска для минимизации f(x) = x^T A x
    
    Параметры:
        x0       : начальная точка (numpy array, shape (2,))
        lr       : шаг обучения (learning rate)
        tol      : порог по норме градиента для остановки
        max_iter : максимальное число итераций
        verbose  : выводить прогресс?
    
    Возвращает:
        x_hist   : список всех точек x_k (для визуализации)
        f_hist   : список значений f(x_k)
    """
    x = np.array(x0, dtype=float)
    x_hist = [x.copy()]
    f_hist = [function(x)]
    
    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        
        if verbose and k % 50 == 0:
            print(f"Iter {k:4d} | x = {x} | f(x) = {f_hist[-1]:.6e} | ||∇f|| = {grad_norm:.2e}")
        
        if grad_norm < tol:
            if verbose:
                print(f"✅ Сходимость достигнута на итерации {k}. ||∇f|| = {grad_norm:.2e} < {tol}")
            break
        
        # Шаг градиентного спуска: x_{k+1} = x_k - lr * ∇f(x_k)
        x = x - lr * grad
        x_hist.append(x.copy())
        f_hist.append(function(x))
    else:
        if verbose:
            print("⚠️ Достигнуто максимальное число итераций. Сходимость не достигнута.")
    
    return np.array(x_hist), np.array(f_hist)

# Пример запуска
if __name__ == "__main__":
    x0 = np.array([2.0, -1.0])  # начальная точка
    x_hist, f_hist = gradient_descent(x0, lr=0.4, tol=1e-8, max_iter=500)

    x_opt = x_hist[-1]
    print("\n🏁 Результат:")
    print(f"Найденный минимум: x* = {x_opt}")
    print(f"f(x*) = {function(x_opt):.2e}")
    print("Точный минимум: x* = [0, 0] (т.к. A положительно определена)")