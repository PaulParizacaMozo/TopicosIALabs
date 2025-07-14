# Versión mínima de CMake requerida.
cmake_minimum_required(VERSION 3.10)

# --- Información del Proyecto ---
# Se establece el nombre del proyecto como "ViT" y el lenguaje como C++.
project(ViT CXX)

# --- Estándar del Lenguaje ---
# Se exige el estándar C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Configuración de Tipos de Compilación y Flags ---

# Establece "Release" como el tipo de compilación por defecto si no se especifica.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type: Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
endif()

# Flags de compilación comunes para todos los modos (buenas prácticas).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

# Flags específicos por tipo de compilación.
# -O3: Máxima optimización para velocidad. -DNDEBUG: Desactiva las aserciones.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# -g: Incluye símbolos de depuración. -O0: Desactiva todas las optimizaciones.
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

# --- Búsqueda de Dependencias ---
# Busca el paquete OpenMP para la paralelización. Es requerido para compilar.
find_package(OpenMP REQUIRED)

# --- Configuración de Directorios ---
# Añade el directorio 'include' a las rutas de búsqueda de cabeceras.
# Esto permite hacer #include "core/Tensor.hpp" en lugar de #include "include/core/Tensor.hpp".
include_directories(include)

# Ubica el ejecutable de salida en un directorio 'bin' dentro del directorio de compilación.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Recopilación de Archivos Fuente ---
# Encuentra automáticamente todos los archivos .cpp dentro del directorio 'src'.
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
)

# Define explícitamente el archivo principal de la aplicación.
set(MAIN_SOURCE "app/main.cpp")

# --- Creación del Ejecutable ---
# Crea un ejecutable llamado "ViT" a partir de todos los archivos fuente recopilados.
add_executable(${PROJECT_NAME} ${MAIN_SOURCE} ${SOURCES})

# --- Enlace de Librerías ---
# Enlaza OpenMP a nuestro ejecutable.
if(OpenMP_FOUND)
    message(STATUS "OpenMP encontrado, enlazando...")
    # La forma moderna y recomendada de enlazar OpenMP.
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP no se encontró. La compilación continuará sin paralelización.")
endif()

# Mensaje final de configuración
message(STATUS "Configuración de CMake para ${PROJECT_NAME} completada.")
