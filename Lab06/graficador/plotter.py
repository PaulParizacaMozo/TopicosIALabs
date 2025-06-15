
import pandas as pd
import matplotlib.pyplot as plt

# Función para leer varios archivos y graficar los resultados de la prueba y entrenamiento
def graficar_resultados_comparativos(archivos):
    # Preparar un diccionario para almacenar los datos de cada método
    metodos = []
    perdidas_train = []
    perdidas_test = []
    precisiones_train = []
    precisiones_test = []
    
    # Leer los archivos y almacenar los datos
    for archivo, metodo in archivos.items():
        # Leer el archivo CSV
        df = pd.read_csv(archivo)
        
        # Almacenar las pérdidas y precisiones de train y test
        metodos.append(metodo)
        perdidas_train.append(df['Perdida_Train'])
        perdidas_test.append(df['Perdida_Test'])
        precisiones_train.append(df['Precision_Train'])
        precisiones_test.append(df['Precision_Test'])

    # Crear el gráfico de la pérdida (train y test)
    plt.figure(figsize=(12, 6))
    
    # Graficar las pérdidas de train y test
    for i, metodo in enumerate(metodos):
        # Colores más claros para Train y más oscuros para Test
        plt.plot(df['Epoca'], perdidas_train[i], label=f'Pérdida Train ({metodo})', linestyle='--', color=f'C{i}', alpha=0.5)
        plt.plot(df['Epoca'], perdidas_test[i], label=f'Pérdida Test ({metodo})', color=f'C{i}')
    
    plt.title('Pérdida Train y Test vs Época para diferentes métodos')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Crear el gráfico de la precisión (train y test)
    plt.figure(figsize=(12, 6))
    
    # Graficar las precisiones de train y test
    for i, metodo in enumerate(metodos):
        # Colores más claros para Train y más oscuros para Test
        plt.plot(df['Epoca'], precisiones_train[i], label=f'Precisión Train ({metodo})', linestyle='--', color=f'C{i}', alpha=0.5)
        plt.plot(df['Epoca'], precisiones_test[i], label=f'Precisión Test ({metodo})', color=f'C{i}')
    
    plt.title('Precisión Train y Test vs Época para diferentes métodos')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Mapeo de archivos largos a nombres 
archivos = {
    'historial_entrenamiento_50_rmsprop.txt': 'RMSprop',
    'historial_entrenamiento_50_adam.txt': 'Adam',
    'historial_entrenamiento_50_adamL2.txt': 'AdamL2',
    'historial_entrenamiento_50_adamDropout.txt': 'Adam+Dropout',
}

# Llamar a la función pasando los archivos
graficar_resultados_comparativos(archivos)
