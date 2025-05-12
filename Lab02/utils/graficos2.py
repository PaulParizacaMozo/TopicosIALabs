import matplotlib.pyplot as plt
import numpy as np

# Datos de los puntos AND para visualización
puntos_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
salidas_and = np.array([0, 0, 0, 1])

# Separar los puntos según su clase para colorearlos (para AND)
ceros_x_and = puntos_and[salidas_and == 0][:, 0]
ceros_y_and = puntos_and[salidas_and == 0][:, 1]
unos_x_and = puntos_and[salidas_and == 1][:, 0]
unos_y_and = puntos_and[salidas_and == 1][:, 1]

# --- Recta 1: Generada por tu código (función escalón) ---
peso_x_escalon = 0.256272
peso_y_escalon = 0.262401
bias_escalon = -0.437924

# Ecuación de la recta 1: peso_x * x + peso_y * y + bias = 0
# Despejando y: y = (-peso_x * x - bias) / peso_y
x_escalon = np.linspace(-0.5, 2.5, 100)  # Extendemos el límite superior de x
y_escalon = (-peso_x_escalon * x_escalon - bias_escalon) / peso_y_escalon

# --- Recta 2: Generada por las librerías de Python ---
pesos_python = np.array([0.2, 0.1])
bias_python = np.array([-0.2])

# Ecuación de la recta 2: pesos[0] * x + pesos[1] * y + bias[0] = 0
# Despejando y: y = (-pesos[0] * x - bias[0]) / pesos[1]
x_python = np.linspace(-0.5, 2.5, 100)  # Extendemos el límite superior de x
y_python = (-pesos_python[0] * x_python - bias_python[0]) / pesos_python[1]

# --- Crear la gráfica ---
plt.figure(figsize=(8, 8))  # Aumentamos el tamaño para mejor visualización

# Graficar los puntos AND
plt.scatter(ceros_x_and, ceros_y_and, color='blue', marker='o', label='Salida 0 (AND)')
plt.scatter(unos_x_and, unos_y_and, color='green', marker='x', label='Salida 1 (AND)')

# Graficar la Recta 1 (función escalón)
plt.plot(x_escalon, y_escalon, color='purple', linestyle='-', label='Recta Escalon')

# Graficar la Recta 2 (librerías de Python)
plt.plot(x_python, y_python, color='orange', linestyle='--', label='Recta Librerías Python')

# Establecer límites de los ejes para que se vea encuadrado
plt.xlim(-0.5, 2.5)  # Límite superior de x extendido
plt.ylim(-0.5, 2)

# Añadir etiquetas y título
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rectas de Decisión para la Función AND')
plt.legend()
plt.grid(True)
plt.axvline(0, color='black', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.5)

# Mostrar la gráfica
plt.show()
