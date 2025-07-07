import matplotlib.pyplot as plt
import re

# Datos proporcionados para graficar
data_string = """
Epoca  1/20 - Tiempo: 45.99s | train_loss: 1.5933 | train_acc: 0.4058 | val_loss: 0.8344 | val_acc: 0.6899
Epoca  2/20 - Tiempo: 45.42s | train_loss: 0.7166 | train_acc: 0.7351 | val_loss: 0.6574 | val_acc: 0.7631
Epoca  3/20 - Tiempo: 45.09s | train_loss: 0.6304 | train_acc: 0.7675 | val_loss: 0.6074 | val_acc: 0.7816
Epoca  4/20 - Tiempo: 46.14s | train_loss: 0.5950 | train_acc: 0.7819 | val_loss: 0.5768 | val_acc: 0.7944
Epoca  5/20 - Tiempo: 45.76s | train_loss: 0.5629 | train_acc: 0.7930 | val_loss: 0.5499 | val_acc: 0.8097
Epoca  6/20 - Tiempo: 46.07s | train_loss: 0.5404 | train_acc: 0.8037 | val_loss: 0.5323 | val_acc: 0.8175
Epoca  7/20 - Tiempo: 45.03s | train_loss: 0.5252 | train_acc: 0.8097 | val_loss: 0.5050 | val_acc: 0.8242
Epoca  8/20 - Tiempo: 45.66s | train_loss: 0.5107 | train_acc: 0.8154 | val_loss: 0.4851 | val_acc: 0.8299
Epoca  9/20 - Tiempo: 45.22s | train_loss: 0.4941 | train_acc: 0.8223 | val_loss: 0.4767 | val_acc: 0.8338
Epoca 10/20 - Tiempo: 45.37s | train_loss: 0.4876 | train_acc: 0.8250 | val_loss: 0.4760 | val_acc: 0.8347
Epoca 11/20 - Tiempo: 46.03s | train_loss: 0.4857 | train_acc: 0.8250 | val_loss: 0.4647 | val_acc: 0.8365
Epoca 12/20 - Tiempo: 45.75s | train_loss: 0.4907 | train_acc: 0.8244 | val_loss: 0.4588 | val_acc: 0.8357
Epoca 13/20 - Tiempo: 44.82s | train_loss: 0.5143 | train_acc: 0.8169 | val_loss: 0.4693 | val_acc: 0.8316
Epoca 14/20 - Tiempo: 45.65s | train_loss: 0.5146 | train_acc: 0.8147 | val_loss: 0.4806 | val_acc: 0.8256
Epoca 15/20 - Tiempo: 45.83s | train_loss: 0.4837 | train_acc: 0.8230 | val_loss: 0.4758 | val_acc: 0.8283
Epoca 16/20 - Tiempo: 45.85s | train_loss: 0.4756 | train_acc: 0.8261 | val_loss: 0.4677 | val_acc: 0.8290
Epoca 17/20 - Tiempo: 45.21s | train_loss: 0.4678 | train_acc: 0.8287 | val_loss: 0.4559 | val_acc: 0.8315
Epoca 18/20 - Tiempo: 45.19s | train_loss: 0.4642 | train_acc: 0.8310 | val_loss: 0.4522 | val_acc: 0.8337
Epoca 19/20 - Tiempo: 45.83s | train_loss: 0.4581 | train_acc: 0.8329 | val_loss: 0.4496 | val_acc: 0.8359
Epoca 20/20 - Tiempo: 45.69s | train_loss: 0.4583 | train_acc: 0.8324 | val_loss: 0.4507 | val_acc: 0.8368
"""

# Listas (vectores) para almacenar los datos numéricos
epochs = []
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# Expresión regular para extraer los valores de cada línea:
# Busca específicamente los números flotantes/enteros después de sus etiquetas clave.
pattern = re.compile(
    r"Epoca\s+(\d+)/\d+.*?"  # Captura el número de la época (\d+)
    r"train_loss:\s*(\d+\.\d+)\s*\|"  # Captura train_loss (\d+\.\d+)
    r"\s*train_acc:\s*(\d+\.\d+)\s*\|" # Captura train_acc
    r"\s*val_loss:\s*(\d+\.\d+)\s*\|"  # Captura val_loss
    r"\s*val_acc:\s*(\d+\.\d+)"       # Captura val_acc
)

# Procesar cada línea de los datos de entrada
print("--- Iniciando preprocesamiento de datos ---")
for i, line in enumerate(data_string.strip().split('\n')):
    if not line.strip(): # Saltar líneas completamente vacías
        print(f"Línea {i+1}: Vacía, omitiendo.")
        continue
    
    match = pattern.search(line)
    if match:
        try:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            train_acc.append(float(match.group(3)))
            val_loss.append(float(match.group(4)))
            val_acc.append(float(match.group(5)))
            # print(f"Línea {i+1}: Datos extraídos correctamente.") # Puedes descomentar para depuración
        except ValueError as e:
            print(f"Línea {i+1}: Error al convertir un valor numérico: '{line}' - {e}")
    else:
        print(f"Línea {i+1}: No coincide con el patrón esperado y será omitida: '{line}'")

print("--- Preprocesamiento de datos finalizado ---")

# --- Verificación de la data extraída (opcional) ---
print("\n--- Verificación de los Datos Extraídos ---")
if epochs:
    print(f"Total de épocas procesadas: {len(epochs)}")
    print(f"Épocas (primeros 5): {epochs[:5]}")
    print(f"Train Loss (primeros 5): {[f'{x:.4f}' for x in train_loss[:5]]}")
    print(f"Train Accuracy (primeros 5): {[f'{x:.4f}' for x in train_acc[:5]]}")
    print(f"Val Loss (primeros 5): {[f'{x:.4f}' for x in val_loss[:5]]}")
    print(f"Val Accuracy (primeros 5): {[f'{x:.4f}' for x in val_acc[:5]]}")
else:
    print("¡Advertencia! No se extrajeron datos de ninguna época. Por favor, verifica el formato de entrada o el patrón regex.")
print("------------------------------------------")

# --- Código de Graficación Matplotlib ---
# Asegúrate de que haya datos para graficar
if not epochs:
    print("\nNo hay datos para graficar. Asegúrate de que el preprocesamiento haya extraído los valores correctamente.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7)) # 1 fila, 2 columnas. Tamaño ajustado para visibilidad.

    # Gráfico de Loss
    axes[0].plot(epochs, train_loss, label='Train Loss', color='blue', marker='o', markersize=4, linewidth=1.5)
    axes[0].plot(epochs, val_loss, label='Validation Loss', color='red', marker='x', markersize=4, linewidth=1.5)
    axes[0].set_title('Curvas de Loss (Entrenamiento vs. Validación)', fontsize=14)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_xticks(epochs[::2]) # Muestra las épocas de 2 en 2 para un eje X más limpio

    # Gráfico de Accuracy
    axes[1].plot(epochs, train_acc, label='Train Accuracy', color='green', marker='o', markersize=4, linewidth=1.5)
    axes[1].plot(epochs, val_acc, label='Validation Accuracy', color='purple', marker='x', markersize=4, linewidth=1.5)
    axes[1].set_title('Curvas de Accuracy (Entrenamiento vs. Validación)', fontsize=14)
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_xticks(epochs[::2]) # Muestra las épocas de 2 en 2 para un eje X más limpio
    axes[1].set_ylim(0, 1) # Asegura que el rango del Accuracy sea de 0 a 1

    # Ajustar el diseño general y añadir un título principal a la figura
    fig.suptitle('Métricas de Entrenamiento y Validación del Modelo', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Ajusta el layout para que el título general no se solape
    
    # Muestra los gráficos
    plt.show()
