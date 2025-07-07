import matplotlib.pyplot as plt
import re

# Datos proporcionados para graficar
data_string = """
Epoca  1/20 - Tiempo: 7.09s | train_loss: 0.7889 | train_acc: 0.7418 | val_loss: 0.5556 | val_acc: 0.8245
Epoca  2/20 - Tiempo: 7.30s | train_loss: 0.5116 | train_acc: 0.8235 | val_loss: 0.4801 | val_acc: 0.8402
Epoca  3/20 - Tiempo: 7.27s | train_loss: 0.4640 | train_acc: 0.8381 | val_loss: 0.4478 | val_acc: 0.8485
Epoca  4/20 - Tiempo: 8.03s | train_loss: 0.4372 | train_acc: 0.8475 | val_loss: 0.4263 | val_acc: 0.8538
Epoca  5/20 - Tiempo: 7.18s | train_loss: 0.4178 | train_acc: 0.8537 | val_loss: 0.4100 | val_acc: 0.8590
Epoca  6/20 - Tiempo: 7.53s | train_loss: 0.4021 | train_acc: 0.8594 | val_loss: 0.3978 | val_acc: 0.8619
Epoca  7/20 - Tiempo: 7.26s | train_loss: 0.3891 | train_acc: 0.8637 | val_loss: 0.3875 | val_acc: 0.8660
Epoca  8/20 - Tiempo: 7.46s | train_loss: 0.3776 | train_acc: 0.8676 | val_loss: 0.3787 | val_acc: 0.8674
Epoca  9/20 - Tiempo: 8.21s | train_loss: 0.3675 | train_acc: 0.8706 | val_loss: 0.3713 | val_acc: 0.8694
Epoca 10/20 - Tiempo: 7.28s | train_loss: 0.3586 | train_acc: 0.8736 | val_loss: 0.3646 | val_acc: 0.8707
Epoca 11/20 - Tiempo: 7.29s | train_loss: 0.3504 | train_acc: 0.8766 | val_loss: 0.3584 | val_acc: 0.8726
Epoca 12/20 - Tiempo: 7.32s | train_loss: 0.3430 | train_acc: 0.8793 | val_loss: 0.3530 | val_acc: 0.8741
Epoca 13/20 - Tiempo: 7.32s | train_loss: 0.3361 | train_acc: 0.8813 | val_loss: 0.3483 | val_acc: 0.8750
Epoca 14/20 - Tiempo: 7.38s | train_loss: 0.3297 | train_acc: 0.8829 | val_loss: 0.3431 | val_acc: 0.8757
Epoca 15/20 - Tiempo: 7.20s | train_loss: 0.3236 | train_acc: 0.8851 | val_loss: 0.3394 | val_acc: 0.8772
Epoca 16/20 - Tiempo: 7.35s | train_loss: 0.3181 | train_acc: 0.8870 | val_loss: 0.3359 | val_acc: 0.8772
Epoca 17/20 - Tiempo: 7.31s | train_loss: 0.3128 | train_acc: 0.8890 | val_loss: 0.3328 | val_acc: 0.8790
Epoca 18/20 - Tiempo: 7.26s | train_loss: 0.3077 | train_acc: 0.8905 | val_loss: 0.3297 | val_acc: 0.8803
Epoca 19/20 - Tiempo: 7.36s | train_loss: 0.3030 | train_acc: 0.8921 | val_loss: 0.3264 | val_acc: 0.8809
Epoca 20/20 - Tiempo: 7.34s | train_loss: 0.2984 | train_acc: 0.8934 | val_loss: 0.3241 | val_acc: 0.8807
"""

# Vectores para almacenar los datos numéricos
epochs = []
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# Expresión regular para extraer los valores
pattern = re.compile(
    r"Epoca\s+(\d+)/\d+.*?"  # Captura el número de la época
    r"train_loss:\s*(\d+\.\d+)\s*\|"  # Captura train_loss
    r"\s*train_acc:\s*(\d+\.\d+)\s*\|" # Captura train_acc
    r"\s*val_loss:\s*(\d+\.\d+)\s*\|"  # Captura val_loss
    r"\s*val_acc:\s*(\d+\.\d+)"       # Captura val_acc
)

# Procesar cada línea para extraer los datos
for line in data_string.strip().split('\n'):
    if not line.strip():
        continue # Ignorar líneas vacías
    
    match = pattern.search(line)
    if match:
        try:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            train_acc.append(float(match.group(3)))
            val_loss.append(float(match.group(4)))
            val_acc.append(float(match.group(5)))
        except ValueError as e:
            print(f"Error al convertir datos numéricos en la línea: '{line}' - {e}")
    else:
        print(f"Línea no coincide con el patrón y fue omitida: '{line}'")

# --- Código de Graficación Matplotlib ---
if not epochs:
    print("\nNo hay datos válidos para graficar. El proceso de extracción de datos no encontró resultados.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7)) # Aumentado el figsize para mejor visualización

    # Gráfico de Loss
    axes[0].plot(epochs, train_loss, label='Train Loss', color='blue', marker='o', markersize=4, linewidth=1.5)
    axes[0].plot(epochs, val_loss, label='Validation Loss', color='red', marker='x', markersize=4, linewidth=1.5)
    axes[0].set_title('Curvas de Loss (Entrenamiento vs. Validación)', fontsize=14)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_xticks(epochs[::2]) # Mostrar solo cada 2 épocas en el eje X para no saturar

    # Gráfico de Accuracy
    axes[1].plot(epochs, train_acc, label='Train Accuracy', color='green', marker='o', markersize=4, linewidth=1.5)
    axes[1].plot(epochs, val_acc, label='Validation Accuracy', color='purple', marker='x', markersize=4, linewidth=1.5)
    axes[1].set_title('Curvas de Accuracy (Entrenamiento vs. Validación)', fontsize=14)
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_xticks(epochs[::2]) # Mostrar solo cada 2 épocas en el eje X para no saturar
    axes[1].set_ylim(0, 1) # El accuracy va de 0 a 1

    # Ajustar el diseño y añadir un título general a la figura
    fig.suptitle('Métricas de Entrenamiento y Validación del Modelo MLP', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Ajustar layout para hacer espacio al supertítulo
    
    # Muestra los gráficos
    plt.show()
