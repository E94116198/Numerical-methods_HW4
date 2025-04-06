import numpy as np

# Composite Simpson's Rule
def composite_simpson(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    S = y[0] + y[-1]  # f(x0) + f(xn)
    for i in range(1, n):
        if i % 2 == 0:
            S += 2 * y[i]  # even index
        else:
            S += 4 * y[i]  # odd index

    return (h / 3) * S

# Problem (a): ∫₀¹ x^(-1/4) sin(x) dx
# Use substitution: t = 1/x → x = 1/t, dx = -1/t^2 dt
# Bounds: x = 0 → t = ∞, x = 1 → t = 1
# So integral becomes: ∫₁^∞ t^(-7/4) * sin(1/t) dt

def f_a(t):
    return t ** (-7/4) * np.sin(1 / t)

# Approximate upper bound for ∞
a_a, b_a = 1, 100
n = 4
result_a = composite_simpson(f_a, a_a, b_a, n)

# Problem (b): ∫₁^∞ x^(-4) sin(x) dx
# Use substitution: t = 1/x → x = 1/t, dx = -1/t^2 dt
# Bounds: x = 1 → t = 1, x = ∞ → t = 0
# So integral becomes: ∫₀¹ t^2 * sin(1/t) dt

def f_b(t):
    return t**2 * np.sin(1 / t)

# Avoid division by zero at t = 0, so we integrate from 0.01
a_b, b_b = 0.01, 1
result_b = composite_simpson(f_b, a_b, b_b, n)

# Output the results
print("Approximation for (a):", result_a)
print("Approximation for (b):", result_b)
