#!/usr/bin/env python3
"""
Parche para madmom: compatibilidad con Python 3.10+ y NumPy 1.20+
"""
import re
import sys
from pathlib import Path

def patch_file(file_path):
    """Aplica parches a un archivo de madmom"""
    try:
        content = file_path.read_text()
        original_content = content

        # Parche 1: MutableSequence de collections a collections.abc
        content = content.replace(
            'from collections import MutableSequence',
            'from collections.abc import MutableSequence'
        )

        # Parche 2: basestring no existe en Python 3
        # Reemplazar: string_types = basestring
        # Por: string_types = str
        content = re.sub(
            r'\bstring_types\s*=\s*basestring\b',
            'string_types = str',
            content
        )

        # Parche 3: np.float -> np.float64 (pero no np.float32, np.float64, etc)
        content = re.sub(
            r'\bnp\.float\b(?![\d_])',
            'np.float64',
            content
        )

        # Parche 4: np.int -> np.int64 (pero no np.int32, np.int64, np.integer, etc)
        content = re.sub(
            r'\bnp\.int\b(?![\d_a-z])',
            'np.int64',
            content
        )

        # Parche 5: np.bool -> np.bool_ (pero no np.bool8, np.bool_, etc)
        content = re.sub(
            r'\bnp\.bool\b(?![\d_])',
            'np.bool_',
            content
        )

        # Solo escribir si hubo cambios
        if content != original_content:
            file_path.write_text(content)
            print(f"Parcheado: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error al parchear {file_path}: {e}", file=sys.stderr)
        return False

def main():
    # Buscar todos los archivos .py en madmom
    madmom_path = Path('/usr/local/lib/python3.10/site-packages/madmom')
    if not madmom_path.exists():
        print(f"Error: {madmom_path} no existe", file=sys.stderr)
        sys.exit(1)

    patched_count = 0
    for py_file in madmom_path.rglob('*.py'):
        if patch_file(py_file):
            patched_count += 1

    print(f"\nTotal de archivos parcheados: {patched_count}")

if __name__ == '__main__':
    main()
