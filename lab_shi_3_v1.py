import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import sys

# Вимкнення попереджень TensorFlow (опціонально)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Встановлення кодування для stdout (для Windows, якщо є проблеми з кирилицею)
if sys.platform == "win32":
    os.system('chcp 65001 > nul')


# Встановлення фіксованого seed для відтворюваності
def set_seeds(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)


# --- 1. Налаштування задачі (Варіант 7) ---
alpha = 1.0
xmin, xmax = 0.0, 1.0
tmin, tmax = 0.0, 1.0


# Початкова умова (IC): u(x,0) = lg(x+1) + 1.3
def u_ic(x_tensor):
    # Переконуємось, що x_tensor - це numpy array для log10,
    # потім конвертуємо назад в tf.Tensor
    x_np = x_tensor.numpy()
    log_vals = np.log10(x_np + 1.0) + 1.3
    return tf.constant(log_vals, dtype=tf.float32)


# Граничні умови (BC)
def u_bc_left(t):
    return t ** 2 + 1.3


def u_bc_right(t):
    return t ** 2 + 1.6010


# --- 2. Генерація колокаційних точок ---
def get_collocation_points(N_f=10000, N_b=1000, N_0=1000):
    # PDE точки (внутрішні)
    x_f = tf.random.uniform((N_f, 1), xmin, xmax)
    t_f = tf.random.uniform((N_f, 1), tmin, tmax)

    # BC точки (на межах)
    t_b = tf.random.uniform((N_b, 1), tmin, tmax)
    x_bL = tf.zeros_like(t_b) + xmin
    x_bR = tf.zeros_like(t_b) + xmax

    # IC точки (початкові)
    x_0 = tf.random.uniform((N_0, 1), xmin, xmax)
    t_0 = tf.zeros_like(x_0) + tmin

    # Цільові значення для IC
    u_0_target = u_ic(x_0)

    # Цільові значення для BC
    u_bL_target = u_bc_left(t_b)
    u_bR_target = u_bc_right(t_b)

    points = {
        'x_f': x_f, 't_f': t_f,
        'x_bL': x_bL, 't_b_L': t_b, 'u_bL_target': u_bL_target,
        'x_bR': x_bR, 't_b_R': t_b, 'u_bR_target': u_bR_target,
        'x_0': x_0, 't_0': t_0, 'u_0_target': u_0_target
    }
    return points


# Допоміжна функція для об'єднання входів
def to_input(x, t):
    return tf.concat([x, t], axis=1)


# --- 3. Модель PINN ---
def build_model(width=128, depth=6, activation='tanh'):
    if activation == 'swish':
        act_func = tf.nn.swish
    else:
        act_func = tf.nn.tanh

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(2,)))
    for _ in range(depth):
        model.add(tf.keras.layers.Dense(width, activation=act_func,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal()))
    model.add(tf.keras.layers.Dense(1, activation=None))
    return model


# --- 4. Обчислення залишків (AutoDiff) ---
@tf.function
def pde_residual(model, x, t):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, t])
            # Модель очікує вхід форми (N, 2)
            u_input = to_input(x, t)
            u = model(u_input)
        u_x = tape1.gradient(u, x)
        u_t = tape1.gradient(u, t)

    # Перевірка, чи u_x не None (може трапитись, якщо x не 'watched')
    if u_x is None:
        # Це не повинно траплятися з watch, але це захист
        u_xx = tf.zeros_like(u)
    else:
        u_xx = tape2.gradient(u_x, x)
        if u_xx is None:  # Якщо u_x не має градієнта по x
            u_xx = tf.zeros_like(u)

    del tape1, tape2

    residual = u_t - alpha * u_xx
    return u, residual


# --- 5. Функція втрат ---
@tf.function
def get_loss(model, points):
    # 1. Втрати PDE (залишок рівняння)
    _, R_f = pde_residual(model, points['x_f'], points['t_f'])
    loss_pde = tf.reduce_mean(tf.square(R_f))

    # 2. Втрати BC (граничні умови)
    u_bL, _ = pde_residual(model, points['x_bL'], points['t_b_L'])
    u_bR, _ = pde_residual(model, points['x_bR'], points['t_b_R'])

    loss_bc_L = tf.reduce_mean(tf.square(u_bL - points['u_bL_target']))
    loss_bc_R = tf.reduce_mean(tf.square(u_bR - points['u_bR_target']))
    loss_bc = loss_bc_L + loss_bc_R

    # 3. Втрати IC (початкова умова)
    u_init, _ = pde_residual(model, points['x_0'], points['t_0'])
    loss_ic = tf.reduce_mean(tf.square(u_init - points['u_0_target']))

    # Загальна функція втрат (ваги = 1)
    loss = loss_pde + loss_bc + loss_ic

    return loss, loss_pde, loss_bc, loss_ic


# Обчислення градієнтів (окремо для @tf.function)
@tf.function
def loss_and_grads(model, points):
    with tf.GradientTape() as tape:
        loss, loss_pde, loss_bc, loss_ic = get_loss(model, points)

    grads = tape.gradient(loss, model.trainable_variables)
    return loss, loss_pde, loss_bc, loss_ic, grads


# --- 6. Цикл навчання ---
def train_model(seed, activation_func, epochs=5000, lr=1e-3):
    print(f"\n--- Початок тренування: Seed={seed}, Активація={activation_func} ---")
    set_seeds(seed)

    # Отримуємо точки (щоб вони були однакові для однакових seeds)
    points = get_collocation_points()

    model = build_model(width=128, depth=6, activation=activation_func)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    history = {"loss": [], "pde": [], "bc": [], "ic": []}
    log_every = 500

    start_time = time.time()
    for ep in range(1, epochs + 1):
        loss, lp, lb, li, grads = loss_and_grads(model, points)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if ep % log_every == 0 or ep == 1:
            # Зберігаємо значення
            history["loss"].append(float(loss))
            history["pde"].append(float(lp))
            history["bc"].append(float(lb))
            history["ic"].append(float(li))
            print(f"Epoch {ep:5d}: total={loss:.3e} | pde={lp:.3e} | bc={lb:.3e} | ic={li:.3e}")

    end_time = time.time()
    print(f"Тренування завершено. Час: {end_time - start_time:.2f} сек.")

    final_metrics = {
        "loss": history["loss"][-1],
        "pde": history["pde"][-1],
        "bc": history["bc"][-1],
        "ic": history["ic"][-1]
    }

    return model, history, final_metrics


# --- 7. Візуалізація результатів ---
def visualize_results(model, title_suffix=""):
    nx, nt = 200, 200
    xg = np.linspace(xmin, xmax, nx)
    tg = np.linspace(tmin, tmax, nt)
    Xg, Tg = np.meshgrid(xg, tg)

    x_flat = tf.constant(Xg.reshape(-1, 1), dtype=tf.float32)
    t_flat = tf.constant(Tg.reshape(-1, 1), dtype=tf.float32)

    u_pred, _ = pde_residual(model, x_flat, t_flat)
    u_pred = tf.reshape(u_pred, (nt, nx))

    # Візуалізація теплової карти PINN
    plt.figure(figsize=(6, 5))
    plt.imshow(u_pred.numpy(), origin='lower', aspect='auto',
               extent=[xmin, xmax, tmin, tmax])
    plt.title(f"PINN u(x,t) ({title_suffix})")
    plt.xlabel("x (координата)")
    plt.ylabel("t (час)")
    plt.colorbar(label="u (Температура)")
    plt.tight_layout()
    plt.show()

    # Візуалізація 1D профілів
    plt.figure(figsize=(8, 6))
    for t_sel in [0.0, 0.25, 0.5, 1.0]:
        x_line = tf.linspace(xmin, xmax, 200)[:, None]
        t_line = tf.ones_like(x_line) * t_sel

        u_p, _ = pde_residual(model, x_line, t_line)

        plt.plot(x_line.numpy(), u_p.numpy(), label=f'PINN t={t_sel}', lw=2)

    plt.legend()
    plt.xlabel("x (координата)")
    plt.ylabel("u (Температура)")
    plt.title(f"Профілі u(x,t) у вибрані моменти часу ({title_suffix})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# --- 8. Головна функція запуску ---
def main():
    EPOCHS = 5000  # Можете збільшити до 10000-15000 для кращої точності

    print("=== Завдання 3: Результати базової моделі (tanh, seed=42) ===")
    model_tanh, history_tanh, metrics_tanh = train_model(
        seed=42,
        activation_func='tanh',
        epochs=EPOCHS
    )
    visualize_results(model_tanh, title_suffix="tanh")

    # --- Завдання 7.1: tanh vs Swish ---
    print("\n=== Завдання 7.1: Дослідження tanh vs Swish ===")
    print("Тренування з 'swish' (seed=42)...")
    model_swish, history_swish, metrics_swish = train_model(
        seed=42,
        activation_func='swish',
        epochs=EPOCHS
    )
    visualize_results(model_swish, title_suffix="swish")

    print("\nПорівняння кінцевих втрат (tanh vs Swish):")
    print(
        f"  tanh:  Total={metrics_tanh['loss']:.3e} | PDE={metrics_tanh['pde']:.3e} | BC={metrics_tanh['bc']:.3e} | IC={metrics_tanh['ic']:.3e}")
    print(
        f"  swish: Total={metrics_swish['loss']:.3e} | PDE={metrics_swish['pde']:.3e} | BC={metrics_swish['bc']:.3e} | IC={metrics_swish['ic']:.3e}")

    # Побудова графіків збіжності
    plt.figure(figsize=(10, 6))
    epochs_axis = np.arange(1, EPOCHS + 1, 500)
    plt.plot(epochs_axis, history_tanh['loss'], 'b-o', label='Total Loss (tanh)')
    plt.plot(epochs_axis, history_swish['loss'], 'r-s', label='Total Loss (swish)')
    plt.yscale('log')
    plt.title('Збіжність Total Loss (tanh vs Swish)')
    plt.xlabel('Епохи')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    # --- Завдання 7.2: Перевірка стабільності (4 seeds) ---

    print("\n=== Завдання 7.2: Перевірка стабільності (4 seeds, tanh) ===")
    seeds = [42, 123, 456, 789]
    all_metrics = []

    for s in seeds:
        _, _, metrics = train_model(seed=s, activation_func='tanh', epochs=EPOCHS)
        all_metrics.append(metrics)

    # Збір даних для аналізу
    losses = np.array([m['loss'] for m in all_metrics])
    pde_losses = np.array([m['pde'] for m in all_metrics])
    bc_losses = np.array([m['bc'] for m in all_metrics])
    ic_losses = np.array([m['ic'] for m in all_metrics])

    print("\nРезультати дослідження стабільності (4 seeds):")
    print(f"Seeds: {seeds}")
    print(f"Total Loss: {losses}")

    print("\nСтатистика (Середнє ± СКО):")
    print(f"Total Loss: {np.mean(losses):.3e} ± {np.std(losses):.3e}")
    print(f"PDE Loss:   {np.mean(pde_losses):.3e} ± {np.std(pde_losses):.3e}")
    print(f"BC Loss:    {np.mean(bc_losses):.3e} ± {np.std(bc_losses):.3e}")
    print(f"IC Loss:    {np.mean(ic_losses):.3e} ± {np.std(ic_losses):.3e}")


if __name__ == "__main__":
    main()