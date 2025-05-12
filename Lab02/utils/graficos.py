import matplotlib.pyplot as plt
import numpy as np

# Coeficientes de la ecuación de la recta (los mismos que antes)
coef_x = 0.156272
coef_y = 0.762401
constante = -0.0379238

# Generar puntos para la recta
x = np.linspace(-0.5, 1.5, 100)  # Genera 100 puntos entre -0.5 y 1.5 para el eje x (encuadre)
y = (-coef_x * x - constante) / coef_y

# Datos de los puntos OR para visualización
puntos_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
salidas_or = np.array([0, 1, 1, 1])  # Salidas para la función OR

# Separar los puntos según su clase para colorearlos (para OR)
ceros_x = puntos_or[salidas_or == 0][:, 0]
ceros_y = puntos_or[salidas_or == 0][:, 1]
unos_x = puntos_or[salidas_or == 1][:, 0]
unos_y = puntos_or[salidas_or == 1][:, 1]

# Crear la gráfica
plt.figure(figsize=(6, 6))  # Asegura una figura cuadrada

# Graficar la recta
plt.plot(x, y, color='purple', linestyle='-', label='Recta de Decisión')

# Graficar los puntos OR
plt.scatter(ceros_x, ceros_y, color='blue', marker='o', label='Salida 0 (OR)')
plt.scatter(unos_x, unos_y, color='green', marker='x', label='Salida 1 (OR)')

# Establecer límites de los ejes para que se vea encuadrado
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

# Añadir etiquetas y título
plt.xlabel('x')
plt.ylabel('y')
plt.title('Recta de Decisión para la Función OR (Pesos Actualizados)')
plt.legend()
plt.grid(True)
plt.axvline(0, color='black', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.5)

# Mostrar la gráfica
plt.show()
