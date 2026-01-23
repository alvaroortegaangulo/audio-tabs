"""
Monkey patch para restaurar aliases deprecados de NumPy.
Necesario para compatibilidad con madmom y otras librerías antiguas.

Este módulo debe ser importado ANTES de cualquier import de madmom.
"""
import numpy as np

# Restaurar aliases deprecados que fueron eliminados en NumPy 1.24+
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'str'):
    np.str = np.str_
