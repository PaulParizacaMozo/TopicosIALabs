import numpy as np
from sklearn.linear_model import Perceptron
import time

def perceptron_simple(entradas, salidas_esperadas, tasa_aprendizaje, max_iter, pesos_iniciales, bias_inicial):
    perceptron = Perceptron(max_iter=max_iter, eta0=tasa_aprendizaje)
    perceptron.coef_ = pesos_iniciales
    perceptron.intercept_ = np.array([bias_inicial])
    inicio_entrenamiento = time.time()
    perceptron.fit(entradas, salidas_esperadas)
    fin_entrenamiento = time.time()
    tiempo_entrenamiento_us = int((fin_entrenamiento - inicio_entrenamiento) * 1_000_000)
    return perceptron, tiempo_entrenamiento_us

def probar_funcion_logica(nombre_funcion, entradas, salidas_esperadas, tasa_aprendizaje, max_iter, pesos_iniciales, bias_inicial):
    print(f"\n=== Entrenamiento para la función {nombre_funcion} ===\n")
    perceptron_entrenado, tiempo_entrenamiento = perceptron_simple(
        entradas, salidas_esperadas, tasa_aprendizaje, max_iter, pesos_iniciales, bias_inicial
    )
    print("tasa_aprendizaje:", tasa_aprendizaje)
    print("Pesos iniciales:", pesos_iniciales)
    print("Bias inicial:", bias_inicial)
    print("Tiempo de entrenamiento:", f"{tiempo_entrenamiento} microsegundos")
    print("Pesos aprendidos:", perceptron_entrenado.coef_)
    print("Bias aprendido:", perceptron_entrenado.intercept_)
    predicciones = perceptron_entrenado.predict(entradas)
    print("Predicciones:", predicciones)
    print("Salidas esperadas:", salidas_esperadas)
    print("\nProbando nuevas entradas:")
    pruebas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for i in range(len(pruebas)):
        prediccion = perceptron_entrenado.predict(pruebas[i].reshape(1, -1))[0]
        print(f"Entrada: [{pruebas[i][0]}, {pruebas[i][1]}] -> Predicción: {prediccion}")

if __name__ == "__main__":
    entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    tasa_aprendizaje = 0.1
    max_iter = 1000
    pesos_iniciales = np.array([[0.156272, 0.762401]])
    bias_inicial = 0.462076

    salidas_and = np.array([0, 0, 0, 1])
    probar_funcion_logica("AND", entradas, salidas_and, tasa_aprendizaje, max_iter, pesos_iniciales, bias_inicial)

    salidas_or = np.array([0, 1, 1, 1])
    probar_funcion_logica("OR", entradas, salidas_or, tasa_aprendizaje, max_iter, pesos_iniciales, bias_inicial)
