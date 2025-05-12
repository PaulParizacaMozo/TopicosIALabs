cmake_minimum_required(VERSION 3.5)
project(PerceptronProject)

# Indicar que queremos crear un ejecutable llamado 'perceptron_ejecutable'
add_executable(perceptron_ejecutable main.cpp perceptron.cpp)

# (Opcional) Si quieres especificar flags de compilación, por ejemplo, para activar todas las advertencias y usar el estándar C++17:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
