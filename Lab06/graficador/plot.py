
import pandas as pd
import matplotlib.pyplot as plt

# Función para leer el archivo CSV y generar los gráficos
def graficar_resultados(archivo):
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(archivo)
    
    # Crear el gráfico de la función de pérdida
    plt.figure(figsize=(10, 5))
    
    # Graficar la pérdida de entrenamiento y la pérdida de prueba
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoca'], df['Perdida_Train'], label='Pérdida Train', color='blue')
    plt.plot(df['Epoca'], df['Perdida_Test'], label='Pérdida Test', color='red')
    plt.title('Pérdida vs Época')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Crear el gráfico de la precisión
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoca'], df['Precision_Train'], label='Precisión Train', color='green')
    plt.plot(df['Epoca'], df['Precision_Test'], label='Precisión Test', color='orange')
    plt.title('Precisión vs Época')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    # Mostrar las gráficas
    plt.tight_layout()
    plt.show()

# Llamar a la función pasando el archivo CSV
archivo = 'historial_entrenamiento_50_adamDropout.txt'
graficar_resultados(archivo)
