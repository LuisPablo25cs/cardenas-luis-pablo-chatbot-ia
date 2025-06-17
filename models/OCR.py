import pytesseract
import numpy as np
import cv2
import base64
from pydantic import BaseModel
from PIL import Image
import io
import re
import sqlite3
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ImgBase64(BaseModel):
    base64_str: str



def OCR(tarjeta: ImgBase64): 
    image_data = base64.b64decode(tarjeta.base64_str)
    image = Image.open(io.BytesIO(image_data))
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    texto = extraerTexto(cv_image)
    print(texto)
    info_empleado = descomprimirTexto(texto)
    print(info_empleado)
    nombre = info_empleado['nombre']
    id = info_empleado['id']
    departamento = info_empleado['departamento']
    success = verificarUsuario(nombre, id, departamento)
    return success, info_empleado

def extraerTexto(img: base64):
    config = r'--oem 3 --psm 6 -l spa+eng'
    text = pytesseract.image_to_string(img, config=config)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def descomprimirTexto(txt: str):
    #Con el uso de expresiones regulares podemos dividir el texto si siempre se lo damos de la misma manera
    regex = r"""
        ^(?P<nombre>[a-zA-ZáéíóúÁÉÍÓÚñÑ\s]+?)  # Nombre mientras no sea ID
        (?:ID\s*:\s*|id\s*:\s*|\s+)            # Separador "ID"
        (?P<id>\d+)                             # ID numérico
        (?:\s*[-:]?\s*|\s+)                     # Separador 
        (?P<departamento>[a-zA-Z]+)?            # Departamento
        $""".strip()
    
    match = re.search(regex, txt.strip(), re.VERBOSE | re.IGNORECASE)
    return {
        'nombre': match.group('nombre').strip(),
        'id': match.group('id'),
        'departamento': match.group('departamento')
    }

def verificarUsuario(nombre: str, id: str, departamento: str):
    conn = sqlite3.connect("../db/empresa.db")
    cursor = conn.cursor()
    query = """
    SELECT * FROM empleados
    WHERE nombre = ?
    AND id = ?
    AND departamento = ?
    """
    parametros = (nombre, id, departamento)
    cursor.execute(query, parametros)
    consulta = cursor.fetchone()
    if consulta:
        print(consulta)
        return True
    else:
        print("No se encontró al usuario")
        return False
        

                
    