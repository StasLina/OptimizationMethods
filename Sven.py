import math
import time

def f(x):
    return 2*x**2-12*x

def swann(x0, step=1):
    x_prev = x0-step
    x_next = x0+step

    f_x0 = f(x0)
    f_x_prev = f(x_prev)
    f_x_next = f(x_next) 

    if f_x_prev >= f_x0 <= f_x_next:
        return (x_prev,  x_next)

    elif f_x_prev <= f_x0 >= f_x_next:
        raise ValueError("Функция не унимодальна")
    
    f_asc = f_x_prev <= f_x0 <= f_x_next

    if (f_asc):
        step = -step

    a_or_b_prev = x0
    k = 1
    a_or_b_new = 0
    
    while True:
        a_or_b_new = a_or_b_prev + 2**k * step
        if f(a_or_b_new) > f(a_or_b_prev):
            break
        a_or_b_prev = a_or_b_new
        k+=1

    if f_asc:
        return (a_or_b_new, a_or_b_prev)
    else:
        return (a_or_b_prev, a_or_b_new)

def half_division(a,b, l = 0.01):
    k = 0
    while True:
        # print(f"k={k}")
        L = math.fabs(b-a)
        x = (a + b) / 2
        y = a + L/4
        z = b - L/4

        # интервал a__y___x___z__b
        if f(y) < f(x):
            a = a
            b = x
            x = y
        elif f(z) < f(x):
            a = x
            b = b
            x = z
        else:
            a = y
            b = z
        
        if math.fabs(b -a) < l:
            return x
        
        k+=1

def dihotomia(a,b,eps,l):
    k=0
    while True:
        # print(f'k={k} a={a} b={b}')
        y= (a+b-eps)/2
        z = (a+b+eps)/2
        if f(y)>f(z):
            a=y
        else:
            b=z
        
        if math.fabs(b-a) < l:
            return (a+b)/2
        k+=1


if __name__ == "__main__":
    x0 = 0
    step = 10

    # --- Замер времени для метода Свенна ---
    start = time.perf_counter()
    a, b = swann(x0, step)
    end = time.perf_counter()
    print(f"Метод Свенна: [{a:.6f}, {b:.6f}] за {end - start:.6f} секунд\n")

    # --- Замер времени для метода половинного деления ---
    start = time.perf_counter()
    c1 = half_division(a, b)
    end = time.perf_counter()
    print(f"Половинное деление: x* = {c1:.6f} за {end - start:.6f} секунд\n")

    # --- Замер времени для метода дихотомии ---
    start = time.perf_counter()
    c2 = dihotomia(a, b, 0.001, 0.01)
    end = time.perf_counter()
    print(f"Дихотомия: x* = {c2:.6f} за {end - start:.6f} секунд\n")

    # Точный минимум (для проверки): f(x) = 2x² - 12x → x* = 3
    print(f"Точный минимум: x* = 3.0")
# if __name__ == "__main__":
#   x0 = 0
#   step = 10
#   a, b = swann(x0, step)
#   print(f"{a}, {b}")

#   c1 = half_division(a,b)
#   print(f"c: {c1}") 

#   c2 = dihotomia(a,b, 0.001, 0.01)
#   print(f"c: {c2}")