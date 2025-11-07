import numpy as np

def function(x):
    A = np.array([[2.0, 0.5],
                  [0.5, 1.0]])
    return x @ A @ x

def main():
    # Параметры
    N = 100          # макс. итераций
    M = 5            # макс. попыток на шаг
    alpha = 1.5      # растяжение
    beta = 0.4       # сжатие
    R = 1e-6         # мин. шаг
    t = 1.0          # начальный шаг
    
    x = np.array([10.0, 15.0])
    k = 0

    while k < N and t > R:
        improved = False
        for j in range(M):
            # Генерируем случайное направление
            ksi = np.random.uniform(-1, 1, size=x.shape)
            ksi /= np.linalg.norm(ksi) + 1e-10  # нормализуем
            
            y = x + t * ksi
            if function(y) < function(x):
                # Успех: пробуем растянуть
                z = x + alpha * (y - x)
                if function(z) < function(x):
                    x = z
                    t *= alpha
                else:
                    x = y
                improved = True
                k += 1
                break
        
        if not improved:
            t *= beta  # уменьшаем шаг
            k += 1

        if k % 10 == 0:
            print(f"Iter {k}: x={x}, f(x)={function(x):.6f}, t={t:.6e}")

    print("Result:", x, "f(x) =", function(x))

if __name__ == "__main__":
    main()
    
# # method2.py

# # задаём точку x0
# # alpha [] >= 1
# # betta [] 0..1
# # N = 100 - максимальное число итераций
# # k = 0
# # j = 1 ...M
# # R - минимлальная величина шага
# # M - максимальное количество ошибок
# M = 5
# import random
# import numpy as np

# default_ksi = None

# k = 0
# j = 1

# alpha = 1.5
# betta = 0.4
# # print(f"{vector_x}")
# # print(f"{default_ksi}")
# current_t = 1

# def function(x):
#     # Пример: f(x) = 2*x1^2 + x1*x2 + x2^2
#     # Матрица A (должна быть симметричной!)
#     A = np.array([
#         [2.0, 0.5],
#         [0.5, 1.0]
#     ])
#     return x @ A @ x

# def abs_ksi(ksi):
#     return ksi

# def calc_y(x,t, ksi):
#     t * ksi / abs_ksi(ksi)

# def step3(x):
#     global j
#     return x + default_ksi[j] * current_t

# def is_end():
#     global j
#     global M

#     if j < M:
#         j += 1
#         return False
#     else:
#         if current_t > R:
#             current_t = betta * current_t
#             j = 1
#             return False
#         else:
#             return true

# def generate_ksi(x):
#     len_of_vars = len(vector_x)
#     default_ksi = np.random.uniform(-1, 1, size=(M, len_of_vars))
#     # print(default_ksi[0])
#     # print(step3(vector_x))
#     # print(function(vector_x))

#     norms= np.linalg.norm(default_ksi, axis=1, keepdims=True)
#     epsilon = 1e-10
#     norms = np.where(norms < epsilon, 1.0, norms)
#     normalized_ksi = default_ksi / norms

#     default_ksi = normalized_ksi
#     return default_ksi

# def main():
#     global default_ksi
#     global k

#     print("main")
#     vector_x = [10, 15]
#     vector_x = np.array(vector_x)
#     default_ksi = generate_ksi()
    
#     vector_y =  (vector_x)

#     value_x = function(vector_x)
#     value_y = function(vector_y)

#     if  value_y < value_x:
#         temp_z = x + alpha * (vector_y - vector_x)
        
#         value_x = function(vector_x)
#         value_z = function(temp_z)
        
#         if value_z < value_x:
#            vector_x = temp_z
#            current_t *= alpha
           
#            k+=1
#            while True:
#                 temp_z = x + alpha * (vector_y - vector_x)    
#                 value_x = function(vector_x)
#                 value_z = function(temp_z)
                
#                 if value_z < value_x:
#                     vector_x = temp_z
#                     current_t *= alpha
#                     k+=1
#                 else:
#                     break

#     if is_end():
#         return
            


            






# if __name__ == "__main__":
#     main()
