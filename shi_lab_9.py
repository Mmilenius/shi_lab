import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs  # Latin Hypercube Sampling
import time

# Перевірка доступності GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Використовується пристрій: {device}")


# 1. Визначення архітектури PINN
class PhysicsInformedNN(nn.Module):
    def __init__(self, layers):
        super(PhysicsInformedNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.initialize_weights()

    def forward(self, x):
        # x очікується у формі [N, 2] (x, t)
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))
        # Останній шар без активації
        x = self.layers[-1](x)
        return x

    def initialize_weights(self):
        # Ініціалізація ваг Xavier
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


# 2. Функції втрат (PDE, IC, BC)
def loss_components(model, D, x_pde, t_pde, x_ic, t_ic, c_ic, x_bc, t_bc, c_bc):
    # 1. Втрати за фізикою (PDE Loss)
    # Вмикаємо обчислення градієнтів для вхідних точок PDE
    x_pde.requires_grad_(True)
    t_pde.requires_grad_(True)

    C_pde = model(torch.cat([x_pde, t_pde], dim=1))

    # Обчислення похідних за допомогою torch.autograd.grad
    # dC/dt
    C_t = torch.autograd.grad(
        C_pde, t_pde,
        grad_outputs=torch.ones_like(C_pde),
        create_graph=True,
        retain_graph=True
    )[0]

    # dC/dx
    C_x = torch.autograd.grad(
        C_pde, x_pde,
        grad_outputs=torch.ones_like(C_pde),
        create_graph=True,
        retain_graph=True
    )[0]

    # d2C/dx2
    C_xx = torch.autograd.grad(
        C_x, x_pde,
        grad_outputs=torch.ones_like(C_x),
        create_graph=True,
        retain_graph=True
    )[0]

    # Залишок PDE: f = dC/dt - D * d2C/dx2
    f_pde = C_t - D * C_xx
    loss_pde = torch.mean(f_pde ** 2)

    # 2. Втрати за початковими умовами (Initial Loss)
    C_ic_pred = model(torch.cat([x_ic, t_ic], dim=1))
    loss_ic = torch.mean((C_ic_pred - c_ic) ** 2)

    # 3. Втрати за граничними умовами (Boundary Loss)
    C_bc_pred = model(torch.cat([x_bc, t_bc], dim=1))
    loss_bc = torch.mean((C_bc_pred - c_bc) ** 2)

    return loss_pde, loss_ic, loss_bc


# 3. Підготовка даних та тренування
# --- Зміни для Варіанту 2 ---
D_val = 0.05  # Коеф. дифузії
x_domain = [-1.0, 1.0]  # Межі області x
t_domain = [0.0, 1.0]  # Моделюємо до T=1.0
# ------------------------------

# Кількість точок
N_pde = 10000  # Точки для фізики (collocation points)
N_ic = 1000  # Точки для початкових умов
N_bc = 1000  # Точки для граничних умов

# 1. Генерація точок PDE (всередині області)
pde_samples = lhs(2, N_pde)
x_pde = torch.tensor(pde_samples[:, 0:1] * (x_domain[1] - x_domain[0]) + x_domain[0], dtype=torch.float32).to(device)
t_pde = torch.tensor(pde_samples[:, 1:2] * (t_domain[1] - t_domain[0]) + t_domain[0], dtype=torch.float32).to(device)

# 2. Генерація точок IC (t=0)
x_ic = torch.tensor(lhs(1, N_ic) * (x_domain[1] - x_domain[0]) + x_domain[0], dtype=torch.float32).to(device)
t_ic = torch.zeros_like(x_ic).to(device)

# --- Зміни для Варіанту 2 ---
# C(x, 0) = exp(-x^2)
c_ic = torch.exp(-(x_ic ** 2)).to(device)  #
# ------------------------------

# 3. Генерація точок BC (x=-1 та x=1)
t_bc = torch.tensor(lhs(1, N_bc) * (t_domain[1] - t_domain[0]) + t_domain[0], dtype=torch.float32).to(device)

# C(-1, t) = exp(-1)
x_bc_left = torch.full_like(t_bc, x_domain[0]).to(device)
# C(1, t) = exp(-1)
x_bc_right = torch.full_like(t_bc, x_domain[1]).to(device)

x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
t_bc = torch.cat([t_bc, t_bc], dim=0)

# --- Зміни для Варіанту 2 ---
# c_bc = exp(-1)
c_bc_value = np.exp(-1.0)
c_bc = torch.full_like(x_bc, c_bc_value).to(device)
# ------------------------------

# Ініціалізація моделі
layers = [2, 40, 40, 40, 40, 1]  # 2 входи (x, t), 4 приховані шари, 1 вихід (C)
model = PhysicsInformedNN(layers).to(device)

# Оптимізатор Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Словник для збереження історії втрат
loss_history = {'total': [], 'pde': [], 'ic': [], 'bc': []}
lambda_ic = 100.0  # Вага для IC
lambda_bc = 100.0  # Вага для BC

print("Початок тренування...")
start_time = time.time()
epochs = 15000  # [cite: 49]

for epoch in range(epochs + 1):
    optimizer.zero_grad()

    loss_pde, loss_ic, loss_bc = loss_components(
        model, D_val, x_pde, t_pde, x_ic, t_ic, c_ic, x_bc, t_bc, c_bc
    )

    # Загальна функція втрат
    total_loss = loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc

    total_loss.backward()
    optimizer.step()

    # Збереження історії
    loss_history['total'].append(total_loss.item())
    loss_history['pde'].append(loss_pde.item())
    loss_history['ic'].append(loss_ic.item())
    loss_history['bc'].append(loss_bc.item())

    if epoch % 1000 == 0:  # [cite: 50]
        print(f'Epoch {epoch}: Total Loss = {total_loss.item():.6f}, '
              f'PDE Loss = {loss_pde.item():.6f}, '
              f'IC Loss = {loss_ic.item():.6f}, '
              f'BC Loss = {loss_bc.item():.6f}')

end_time = time.time()
print(f"Тренування завершено за {end_time - start_time:.2f} сек.")

# 4. Візуалізація результатів
# 4.1. Графіки втрат
plt.figure(figsize=(12, 6))
plt.plot(loss_history['total'], label='Загальні втрати (Total)')
plt.plot(loss_history['pde'], label='Втрати PDE (Physics)', linestyle='--')
plt.plot(np.array(loss_history['ic']) * lambda_ic, label=f'Втрати IC (x{lambda_ic})', linestyle=':')
plt.plot(np.array(loss_history['bc']) * lambda_bc, label=f'Втрати BC (x{lambda_bc})', linestyle='-.')
plt.xlabel('Епоха')
plt.ylabel('Втрати (Log scale)')
plt.yscale('log')
plt.title('Історія втрат під час навчання PINN')
plt.legend()
plt.grid(True)
plt.savefig("variant2_losses.png")  # Збереження графіка

# 4.2. 3D візуалізація розв'язку
model.eval()
x_plot = np.linspace(x_domain[0], x_domain[1], 100)
t_plot = np.linspace(t_domain[0], t_domain[1], 100)
X, T = np.meshgrid(x_plot, t_plot)

# Підготовка точок для моделі
xt_pairs = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
xt_tensor = torch.tensor(xt_pairs, dtype=torch.float32).to(device)

with torch.no_grad():
    C_pred = model(xt_tensor).cpu().numpy()

C = C_pred.reshape(X.shape)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, C, cmap='viridis', rstride=1, cstride=1)
ax.set_xlabel('x (простір)')
ax.set_ylabel('t (час)')
ax.set_zlabel('C(x, t) (концентрація)')
# --- Зміни для Варіанту 2 ---
ax.set_title('PINN розв\'язок 1D дифузії ($C(x, 0) = e^{-x^{2}}$)')
# ------------------------------
ax.view_init(30, -120)
plt.savefig("variant2_3d_plot.png")  # Збереження графіка

# 4.3. 2D зрізи для порівняння
plt.figure(figsize=(10, 6))

# t = 0 (порівняння з IC)
C_t0_pred = C[0, :]
# --- Зміни для Варіанту 2 ---
C_t0_analytical = np.exp(-(x_plot ** 2))
plt.plot(x_plot, C_t0_analytical, 'r--', label='Аналітична $C(x, 0) = e^{-x^{2}}$')
# ------------------------------
plt.plot(x_plot, C_t0_pred, 'b-', label='PINN прогноз $C(x, 0)$')

# Інші часові зрізи
plt.plot(x_plot, C[25, :], 'g:', label=f'PINN прогноз $C(x, t={t_plot[25]:.2f})$')
plt.plot(x_plot, C[50, :], 'm:', label=f'PINN прогноз $C(x, t={t_plot[50]:.2f})$')
plt.plot(x_plot, C[99, :], 'k:', label=f'PINN прогноз $C(x, t={t_plot[99]:.2f})$')

plt.xlabel('x')
plt.ylabel('Концентрація C')
plt.title('Порівняння зрізів концентрації в різні моменти часу')
plt.legend()
plt.grid(True)
plt.savefig("variant2_2d_slices.png")  # Збереження графіка

print("\nГрафіки збережено у файли: variant2_losses.png, variant2_3d_plot.png, variant2_2d_slices.png")