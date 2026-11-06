"""Persistencia del historial de Ottoman."""

import datetime
import json
import os
from pathlib import Path
import tempfile


class HistorialTruco:
    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.protegido = False

    def inicializar(self, semilla):
        if self.ruta.exists():
            return
        try:
            with semilla.open(encoding="utf-8") as archivo:
                datos = json.load(archivo)
            self.guardar(datos)
        except (OSError, UnicodeError, ValueError) as error:
            print(f"Error al cargar semilla {semilla}: {error}")

    def cargar(self, valores_iniciales):
        try:
            with self.ruta.open(encoding="utf-8") as archivo:
                datos = json.load(archivo)
            if not isinstance(datos, dict) or not isinstance(datos.get("parametros_ia"), dict) or not isinstance(datos.get("partidas", []), list):
                raise ValueError("formato de historial inválido")
            for clave, valor in valores_iniciales["parametros_ia"].items():
                datos["parametros_ia"].setdefault(clave, valor)
            datos.setdefault("partidas", [])
            return datos
        except FileNotFoundError:
            return valores_iniciales
        except (OSError, UnicodeError, ValueError) as error:
            print(f"Error al cargar historial {self.ruta}: {error}")
            respaldo = self.ruta.with_name(self.ruta.name + "." + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".bak")
            try:
                self.ruta.rename(respaldo)
            except OSError as error_respaldo:
                self.protegido = True
                print(f"Error al respaldar historial {self.ruta}: {error_respaldo}")
            return valores_iniciales

    def guardar(self, datos):
        if self.protegido:
            print(f"Error al guardar historial {self.ruta}: no se pudo respaldar el original")
            return
        temporal = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.ruta.parent, prefix=self.ruta.name + ".", suffix=".tmp", delete=False) as archivo:
                temporal = Path(archivo.name)
                json.dump(datos, archivo, ensure_ascii=False, indent=2)
                archivo.flush()
                os.fsync(archivo.fileno())
            os.replace(temporal, self.ruta)
        except (OSError, ValueError, TypeError) as error:
            print(f"Error al guardar historial {self.ruta}: {error}")
        finally:
            if temporal is not None and temporal.exists():
                try:
                    temporal.unlink()
                except OSError as error:
                    print(f"Error al borrar archivo temporal {temporal}: {error}")
