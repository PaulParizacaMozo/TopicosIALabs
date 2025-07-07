# Versión mínima de CMake
cmake_minimum_required(VERSION 3.10)

# Nombre del proyecto y lenguaje
project(CNN CXX)

# Establecer el estándar de C++ (C++17 es una buena elección moderna)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Configuración de Tipos de Compilación y Flags ---

# Establecer "Release" como el tipo de compilación por defecto si no se especifica otro.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Debug, Release, RelWithDebInfo, MinSizeRel." FORCE)
endif()

# Añadir flags de compilación.
# -O3 es el nivel de optimización más agresivo. Lo aplicamos solo en modo Release.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Flags comunes para todos los modos (buenas prácticas)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

# --- Búsqueda de Dependencias ---
# Encontrar OpenMP para la paralelización
find_package(OpenMP REQUIRED)

# --- Configuración de Directorios ---
# Añadir el directorio 'include' a las rutas de búsqueda de cabeceras.
include_directories(include)

# Ubicar los ejecutables de salida en un directorio 'bin' dentro de 'build'
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Recopilación de Archivos Fuente ---
# Usar GLOB_RECURSE para encontrar automáticamente todos los archivos .cpp
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "app/*.cpp"
)

# --- Creación del Ejecutable ---
add_executable(${PROJECT_NAME} ${SOURCES})

# --- Enlace de Librerías ---
# Enlazar OpenMP a nuestro ejecutable
if(OpenMP_FOUND)
    message(STATUS "OpenMP encontrado, enlazando...")
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP no se encontró. La compilación continuará sin paralelización.")
endif()

# Mensaje final de configuración
message(STATUS "Configuración de CMake para ${PROJECT_NAME} completada.")
