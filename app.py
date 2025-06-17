from fastapi import FastAPI, Depends
import sqlite3
from fastapi import HTTPException
from models.classifier import chatbot
from models.linearRegression import regresion
import pytesseract
from pydantic import BaseModel
from models.OCR import OCR


app = FastAPI()

def connect_db(): 
    conn = sqlite3.connect('/db/empresa.db')
    try: 
        yield conn
    finally: 
        conn.close()

@app.get("/")
def read_root():
    return {"Hola mundo :D"}

@app.post("/chatbot")
async def chatbot_endpoint(pregunta: str):
    print("Pregunta recibida:", pregunta)
    try:
        with sqlite3.connect('db/empresa.db') as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query, params, categoria = chatbot(pregunta)
            if categoria == "prediccion":
                return {"Categoria": categoria,
                        "Prediccion": query}
            print(query, params)
            cursor.execute(query, params)
            consulta = cursor.fetchall()
            return {"Categoria": categoria ,
                     "resultados": [dict(row) for row in consulta] }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ImgBase64(BaseModel):
    base64_str: str

@app.post("/upload-tarjeta")
async def upload_tarjeta(tarjeta: ImgBase64):
    try:
        success, info = OCR(tarjeta)
        if success == True: 
            return {"Se escaneo exitosamente la credencial": info, 
                    "validez:": success}
        else:
            return {"No se encontro al usuario": info, 
                    "validez:": success}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando la imagen: {str(e)}")

@app.post("/predict-salario")
async def predictSalario( edad : int, experiencia_anos : int, departamento : str, nivel_educacion: str):
    print(edad, experiencia_anos, departamento, nivel_educacion)
    try: 
        pred, rsqr = regresion(edad, experiencia_anos, departamento, nivel_educacion)
        print(pred, rsqr)
        pred_s = str(round(float(pred),2))
        rsqr_s = str(round(float(rsqr),2))
        return { "El salario esperado es": pred_s, 
                "con una r2 de ": rsqr_s }
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))
