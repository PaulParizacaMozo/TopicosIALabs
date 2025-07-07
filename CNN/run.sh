#!/bin/bash

# Terminar el script inmediatamente si un comando falla
set -e

# --- Variables ---
BUILD_DIR="build"
PROJECT_NAME="CNN"
BUILD_TYPE="Release" # Usar Release para activar la optimización -O3

# --- Funciones ---
build_project() {
  echo "--- Creando directorio de compilación ---"
  mkdir -p ${BUILD_DIR}

  echo "--- Configurando el proyecto con CMake ---"
  # Entramos al directorio de compilación
  cd ${BUILD_DIR}
  # Ejecutamos cmake para generar los archivos de compilación
  cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..

  echo "--- Compilando el proyecto (con optimización ${BUILD_TYPE}) ---"
  # Usamos cmake --build para ser compatibles con cualquier sistema de compilación (Make, Ninja, etc.)
  # El flag -j intenta usar todos los núcleos disponibles para una compilación más rápida.
  cmake --build . --config ${BUILD_TYPE} -- -j$(nproc 2>/dev/null || echo 1)

  echo "--- Compilación completada ---"
  # Volvemos al directorio raíz
  cd ..
}

run_app() {
  echo "--- Ejecutando la aplicación ---"
  ./${BUILD_DIR}/bin/${PROJECT_NAME}
  echo "--- Ejecución finalizada ---"
}

clean_build() {
  echo "--- Limpiando el directorio de compilación ---"
  if [ -d "${BUILD_DIR}" ]; then
    rm -rf ${BUILD_DIR}
    echo "Directorio '${BUILD_DIR}' eliminado."
  else
    echo "El directorio '${BUILD_DIR}' no existe. Nada que limpiar."
  fi
}

# --- Lógica Principal ---
# Si el primer argumento es "clean", solo limpiamos y salimos.
if [ "$1" == "clean" ]; then
  clean_build
  exit 0
fi

# Flujo por defecto: compilar y luego ejecutar.
build_project
run_app

exit 0
