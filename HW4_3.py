import numpy as np
from sympy import symbols, sin, cos, integrate, lambdify, pi
from numpy.polynomial.legendre import leggauss

# 定義符號變數與函數
x, y = symbols('x y')
f_expr = 2*y*sin(x) + cos(x)**2
f_func = lambdify((x, y), f_expr, modules='numpy')

# ==== (c) 精確值 ====
inner = integrate(f_expr, (y, sin(x), cos(x)))
exact_integral = integrate(inner, (x, 0, pi/4))
exact_value = exact_integral.evalf()

# ==== (a) Simpson's Rule ====
n = 4
m = 4
a = 0
b = np.pi / 4
h = (b - a) / n
x_vals = np.array([a + i * h for i in range(n + 1)])
k_vals = (np.cos(x_vals) - np.sin(x_vals)) / (2 * m)

def simpson_weights(k):
    return np.array([1 if i == 0 or i == 2 * k else 4 if i % 2 == 1 else 2 for i in range(2 * k + 1)])

simpson_total = 0
for i in range(n + 1):
    xi = x_vals[i]
    ki = k_vals[i]
    y0 = np.sin(xi)
    y2m = np.cos(xi)
    y_vals = np.array([y0 + j * ki for j in range(2 * m + 1)])
    f_evals = f_func(xi, y_vals)
    inner = ki / 3 * np.dot(simpson_weights(m), f_evals)
    weight = 1 if i == 0 or i == n else 4 if i % 2 == 1 else 2
    simpson_total += weight * inner
simpson_result = h / 3 * simpson_total

# ==== (b) Gaussian Quadrature ====
n_gauss = 3
m_gauss = 3
gx, gw = leggauss(n_gauss)
gy, gw_y = leggauss(m_gauss)

def gaussian_2d(f, n, m):
    result = 0
    for i in range(n):
        xi = (b - a) / 2 * gx[i] + (b + a) / 2
        wx = gw[i]
        y0 = np.sin(xi)
        y1 = np.cos(xi)
        for j in range(m):
            yj = (y1 - y0) / 2 * gy[j] + (y1 + y0) / 2
            wy = gw_y[j]
            result += wx * wy * f(xi, yj) * (y1 - y0) / 2
    return (b - a) / 2 * result

gaussian_result = gaussian_2d(f_func, n_gauss, m_gauss)

# ==== 輸出結果 ====
print("🔹 (a) Simpson’s Rule 結果 =", simpson_result)
print("🔹 (b) Gaussian Quadrature 結果 =", gaussian_result)
print("🔹 (c) 精確值 =", exact_value)

# 誤差比較
print("➡️ Simpson 誤差 =", abs(simpson_result - exact_value))
print("➡️ Gaussian 誤差 =", abs(gaussian_result - exact_value))
