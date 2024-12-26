import numpy as np
import matplotlib.pyplot as plt

# Преобразования в последовательности 7 -> 52 -> 54 -> 53

def transform_to_im_positive(z):
    return 0.5 * (z + 1 / z)

def inverse_transform_to_horizontal_re_positive_stripe(w):
    return np.arccosh(w)

def inverse_transform_to_vertical_positive_stripe(w):
    return w * 1j/np.pi + 1

def inverse_transform_to_second_graph(w):
    return 2 / (1 - w)

def generate_inverted_domain():
    theta = np.linspace(0.00000001, np.pi, 300)
    r = np.geomspace(0.00000000000000000001, 9000000000000000000, 30000)  # Логарифмическое распределение
    R, T = np.meshgrid(r, theta)

    # преобразование полярных координат в декартовые
    X = R * np.cos(T)
    Y = R * np.sin(T)

    # комплексные координаты
    Z = X + 1j * Y

    # исключаем область |z| <= 1
    mask = (np.abs(Z) > 1)

    return Z[mask]

# исходная область: Im(z) > 0, |z| > 1
Z_fill = generate_inverted_domain()

# преобразования
W1 = transform_to_im_positive(Z_fill)
W2 = inverse_transform_to_horizontal_re_positive_stripe(W1)
W3 = inverse_transform_to_vertical_positive_stripe(W2)
W_final = inverse_transform_to_second_graph(W3)

W_final_reflected = np.concatenate([W_final, -np.conj(W_final)])

fig, ax = plt.subplots(2, 3, figsize=(18, 12))

ax[0, 0].scatter(np.real(Z_fill), np.imag(Z_fill), s=1, color='blue', alpha=0.6)
ax[0, 0].set_title("Исходная область")
ax[0, 0].set_xlim(-3, 3)
ax[0, 0].set_ylim(-1, 3)
ax[0, 0].set_aspect('equal')

ax[0, 1].scatter(np.real(W1), np.imag(W1), s=1, color='green', alpha=0.6)
ax[0, 1].set_title("После transform_to_im_positive")
ax[0, 1].set_xlim(-3, 3)
ax[0, 1].set_ylim(-1, 3)
ax[0, 1].set_aspect('equal')

ax[0, 2].scatter(np.real(W2), np.imag(W2), s=1, color='orange', alpha=0.6)
ax[0, 2].set_title("После inverse_transform_to_horizontal_re_positive_stripe")
ax[0, 2].set_xlim(-3, 3)
ax[0, 2].set_ylim(-1, 3)
ax[0, 2].set_aspect('equal')

ax[1, 0].scatter(np.real(W3), np.imag(W3), s=1, color='purple', alpha=0.6)
ax[1, 0].set_title("После inverse_transform_to_vertical_positive_stripe")
ax[1, 0].set_xlim(-3, 3)
ax[1, 0].set_ylim(-1, 3)
ax[1, 0].set_aspect('equal')

ax[1, 1].scatter(np.real(W_final_reflected), np.imag(W_final_reflected), s=1, color='red', alpha=0.6)
ax[1, 1].set_title("inverse_transform_to_second_graph")
ax[1, 1].set_xlim(-3, 3)
ax[1, 1].set_ylim(-1, 3)
ax[1, 1].set_aspect('equal')

ax[1, 2].axis('off')

plt.tight_layout()
plt.show()
