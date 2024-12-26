import numpy as np
import matplotlib.pyplot as plt

def transform_to_first_graph(z):
    return np.exp(z)

def transform_to_im_and_re_positive(z):
    return (np.pi/1j)*(z/2 - 1)

def transform_to_vertical_stripe(z):
    return (z - 2) / z

def generate_domain():
    theta = np.linspace(0, np.pi, 300)
    r = np.geomspace(0.00000000000000000001, 90000, 30000)  # Логарифмическое распределение
    R, T = np.meshgrid(r, theta)

    # преобразование полярных координат в декартовые
    X = R * np.cos(T)
    Y = R * np.sin(T)

    # комплексные координаты
    Z = X + 1j * Y

    # исключаем области |z-1| <= 1 и |z+1| <= 1
    mask = (np.abs(Z - 1) > 1) & (np.abs(Z + 1) > 1)
    Z_filtered = Z[mask]

    return Z_filtered

# исходная область: Im(z) > 0, |z+1| > 1, |z-1| > 1
Z_fill = generate_domain()

# обратные преобразования
W1 = transform_to_vertical_stripe(Z_fill)
W2 = transform_to_im_and_re_positive(W1)
W3 = transform_to_first_graph(W2)

# Визуализация всех шагов
fig, ax = plt.subplots(2, 3, figsize=(18, 12))

ax[0, 0].scatter(np.real(Z_fill), np.imag(Z_fill), s=1, color='blue', alpha=0.6)
ax[0, 0].set_title("Исходная область")
ax[0, 0].set_xlim(-3, 3)
ax[0, 0].set_ylim(-1, 3)
ax[0, 0].set_aspect('equal')

ax[0, 1].scatter(np.real(W1), np.imag(W1), s=1, color='green', alpha=0.6)
ax[0, 1].set_title("После transform_to_vertical_stripe")
ax[0, 1].set_xlim(-3, 3)
ax[0, 1].set_ylim(-1, 3)
ax[0, 1].set_aspect('equal')

ax[0, 2].scatter(np.real(W2), np.imag(W2), s=1, color='orange', alpha=0.6)
ax[0, 2].set_title("После transform_to_im_and_re_positive")
ax[0, 2].set_xlim(-3, 3)
ax[0, 2].set_ylim(-1, 3)
ax[0, 2].set_aspect('equal')

ax[1, 0].scatter(np.real(W3), np.imag(W3), s=1, color='purple', alpha=0.6)
ax[1, 0].set_title("После transform_to_first_graph")
ax[1, 0].set_xlim(-3, 3)
ax[1, 0].set_ylim(-1, 3)
ax[1, 0].set_aspect('equal')

ax[1, 1].axis('off')
ax[1, 2].axis('off')

plt.tight_layout()
plt.show()
