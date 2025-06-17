import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from models.linearRegression import regresion
import re
#Banco de preguntas para la clasificación. 
preguntas = {
    # Tipo: conteo (25 preguntas)
    "¿Cuántos empleados hay?": "conteo",
    "¿Cuántos trabajadores están en IT?": "conteo",
    "¿Cuántos empleados tienen experiencia mayor a 5 años?": "conteo",
    "¿Cuántas personas trabajan en el departamento de Ventas?": "conteo",
    "¿Cuántos empleados son mayores de 30 años?": "conteo",
    "¿Cuántos empleados tienen un título de Maestría?": "conteo",
    "¿Cuántos empleados viven en Hermosillo?": "conteo",
    "¿Cuántos trabajadores ganan más de 25,000?": "conteo",
    "¿Cuántos empleados fueron contratados en 2024?": "conteo",
    "¿Cuántos empleados tienen menos de 2 años de experiencia?": "conteo",
    "¿Cuántos empleados son del departamento de Finanzas?": "conteo",
    "¿Cuántos empleados tienen Licenciatura?": "conteo",
    "¿Cuántos empleados trabajan en Guadalajara?": "conteo",
    "¿Cuántos empleados son menores de 25 años?": "conteo",
    "¿Cuántos trabajadores tienen doctorado?": "conteo",
    "¿Cuántos empleados ganan menos de 20,000?": "conteo",
    "¿Cuántos empleados fueron contratados después de 2022?": "conteo",
    "¿Cuántos trabajadores tienen experiencia en Marketing?": "conteo",
    "¿Cuántos empleados trabajan a tiempo completo?": "conteo",
    "¿Cuántos empleados tienen menos de 1 año de experiencia?": "conteo",
    "¿Cuántos empleados tienen más de 10 años de experiencia?": "conteo",
    "¿Cuántos empleados trabajan en Monterrey?": "conteo",
    "¿Cuántos empleados ganan exactamente 30,000?": "conteo",
    "¿Cuántos empleados tienen título técnico?": "conteo",
    "¿Cuántos empleados trabajan en el departamento de Recursos Humanos?": "conteo",

    # Tipo: busqueda_max (20 preguntas)
    "¿Quién gana más en el departamento de IT?": "busqueda_max",
    "¿Quién es el empleado con mayor experiencia?": "busqueda_max",
    "¿Cuál es el salario más alto en Finanzas?": "busqueda_max",
    "¿Quién es el empleado más viejo?": "busqueda_max",
    "¿Cuál es la ciudad con el empleado que más gana?": "busqueda_max",
    "¿Quién tiene la mayor antigüedad en la empresa?": "busqueda_max",
    "¿Cuál es el salario máximo en Ventas?": "busqueda_max",
    "¿Quién tiene doctorado y mayor salario?": "busqueda_max",
    "¿Quién es el empleado con mayor experiencia en Marketing?": "busqueda_max",
    "¿Quién es el empleado más viejo en Recursos Humanos?": "busqueda_max",
    "¿Cuál es el mayor salario registrado en la base de datos?": "busqueda_max",
    "¿Quién es el empleado con más años en la empresa?": "busqueda_max",
    "¿Quién es el empleado más senior en IT?": "busqueda_max",
    "¿Cuál es el salario más alto para empleados menores de 30 años?": "busqueda_max",
    "¿Quién tiene el mayor salario en Guadalajara?": "busqueda_max",
    "¿Quién es el empleado con mayor experiencia en la empresa?": "busqueda_max",
    "¿Cuál es el mayor salario entre los empleados": "busqueda_max",
    "¿Quién es el empleado con mayor salario en Monterrey?": "busqueda_max",
    "¿Cuál es el mayor salario para empleados con doctorado?": "busqueda_max",
    "¿Quién es el empleado con mayor salario en ventas?": "busqueda_max",
    "¿Quién tiene el mayor salario en IT con licenciatura?": "busqueda_max",
    "¿Cuál es el salario más alto en Finanzas para empleados con maestría?": "busqueda_max",
    "¿Quién es el empleado con más experiencia en Ventas y licenciatura?": "busqueda_max",
    "¿Cuál es el mayor salario en Marketing para empleados con doctorado?": "busqueda_max",
    "¿Quién gana más en IT con maestría y menos de 5 años de experiencia?": "busqueda_max",
    "¿Cuál es el salario máximo en Finanzas para empleados en Guadalajara?": "busqueda_max",
    "¿Quién es el empleado más antiguo en Ventas con licenciatura?": "busqueda_max",
    "¿Cuál es el mayor salario en Marketing para empleados en Monterrey?": "busqueda_max",
    "¿Quién tiene el salario más alto en IT con doctorado y más de 8 años de experiencia?": "busqueda_max",
    "¿Cuál es el salario máximo en Finanzas para empleados menores de 30 años?": "busqueda_max",
    "¿Quién es el empleado con mayor antigüedad en Ventas y maestría?": "busqueda_max",
    "¿Cuál es el mayor salario en Marketing para empleados con licenciatura?": "busqueda_max",
    "¿Quién gana más en IT con maestría en Hermosillo?": "busqueda_max",
    "¿Cuál es el salario más alto en Finanzas para empleados contratados en 2023?": "busqueda_max",
    "¿Quién es el empleado con más años en la empresa en Marketing y doctorado?": "busqueda_max",


    # Tipo: busqueda_min (20 preguntas)
    "¿Quién es el empleado más joven?": "busqueda_min",
    "¿Cuál es el salario más bajo en el departamento de IT?": "busqueda_min",
    "¿Quién tiene la menor experiencia en Marketing?": "busqueda_min",
    "¿Quién es el empleado con el salario mínimo?": "busqueda_min",
    "¿Cuál es el salario más bajo en Ventas?": "busqueda_min",
    "¿Quién tiene la menor antigüedad en la empresa?": "busqueda_min",
    "¿Quién es el empleado más joven en Finanzas?": "busqueda_min",
    "¿Cuál es el salario mínimo para empleados con título técnico?": "busqueda_min",
    "¿Quién tiene la menor experiencia en IT?": "busqueda_min",
    "¿Quién es el empleado con el salario más bajo en Guadalajara?": "busqueda_min",
    "¿Quién es el empleado más joven en Monterrey?": "busqueda_min",
    "¿Cuál es el salario mínimo para empleados con licenciatura?": "busqueda_min",
    "¿Quién tiene la menor experiencia en Recursos Humanos?": "busqueda_min",
    "¿Quién es el empleado más joven contratado en 2024?": "busqueda_min",
    "¿Cuál es el salario más bajo en la empresa?": "busqueda_min",
    "¿Quién tiene la menor antigüedad en Ventas?": "busqueda_min",
    "¿Quién es el empleado más joven con doctorado?": "busqueda_min",
    "¿Cuál es el salario mínimo entre los empleados mayores de 30 años?": "busqueda_min",
    "¿Quién es el empleado con menor salario en IT?": "busqueda_min",
    "¿Quién tiene la menor experiencia en ventas?": "busqueda_min",
    "¿Quién tiene el menor salario en IT con licenciatura?": "busqueda_min",
    "¿Cuál es el salario más bajo en Finanzas para empleados con maestría?": "busqueda_min",
    "¿Quién es el empleado con menos experiencia en Ventas y licenciatura?": "busqueda_min",
    "¿Cuál es el salario mínimo en Marketing para empleados con doctorado?": "busqueda_min",
    "¿Quién gana menos en IT con maestría y más de 3 años de experiencia?": "busqueda_min",
    "¿Cuál es el salario más bajo en Finanzas para empleados en Monterrey?": "busqueda_min",
    "¿Quién es el empleado más nuevo en Ventas con licenciatura?": "busqueda_min",
    "¿Cuál es el salario mínimo en Marketing para empleados en Guadalajara?": "busqueda_min",
    "¿Quién tiene el salario más bajo en IT con doctorado?": "busqueda_min",
    "¿Cuál es el salario mínimo en Finanzas para empleados mayores de 40 años?": "busqueda_min",
    "¿Quién es el empleado con menos antigüedad en Ventas y maestría?": "busqueda_min",
    "¿Cuál es el salario más bajo en Marketing para empleados con licenciatura?": "busqueda_min", 
    "¿Quién gana menos en IT con maestría en Hermosillo?": "busqueda_min",
    "¿Cuál es el salario mínimo en Finanzas para empleados contratados en 2024?": "busqueda_min",
    "¿Quién es el empleado con menos años en la empresa en Marketing y doctorado?": "busqueda_min",
    
    # Tipo: estadistica (20 preguntas)
    "¿Cuál es el promedio de edad de los empleados?": "estadistica",
    "¿Cuál es el salario promedio en Marketing?": "estadistica",
    "¿Cuál es la mediana del salario en IT?": "estadistica",
    "¿Cuál es el promedio del salario en Finanzas?": "estadistica",
    "¿Cuál es el promedio de experiencia en años de los empleados?": "estadistica",
    "¿Cuál es el promedio de salario para empleados menores de 30 años?": "estadistica",
    "¿Cuál es el promedio de edad en el departamento de Ventas?": "estadistica",
    "¿Cuál es la media de años de experiencia en Recursos Humanos?": "estadistica",
    "¿Cuál es el promedio de salario para empleados con licenciatura?": "estadistica",
    "¿Cuál es la mediana de edad para empleados en IT?": "estadistica",
    "¿Cuál es el promedio de años trabajados en la empresa?": "estadistica",
    "¿Cuál es la mediana de la edad de los empleados?": "estadistica",
    "¿Cuál es el promedio de salario en Guadalajara?": "estadistica",
    "¿Cuál es el promedio de experiencia para empleados con doctorado?": "estadistica",
    "¿Cuál es el promedio de edad para empleados contratados en 2023?": "estadistica",
    "¿Cuál es la media de salario en Monterrey?": "estadistica",
    "¿Cuál es el promedio de experiencia en años para empleados de ventas?": "estadistica",
    "¿Cuál es la mediana de salario para empleados menores de 25 años?": "estadistica",
    "¿Cuál es la ´romedio del salario en marketing?": "estadistica",
    "¿Cuál es el promedio de edad de empleados con título técnico?": "estadistica",

    # Tipo: filtro (20 preguntas)
    "¿Cuántos empleados hay en ventas?": "filtro",
    "¿Cuántos empleados trabajan en IT con experiencia mayor a 5 años?": "filtro",
    "¿Cuántos empleados en Marketing tienen salario mayor a 25,000?": "filtro",
    "¿Cuántos empleados mayores de 30 años trabajan en Finanzas?": "filtro",
    "¿Cuántos empleados en Guadalajara tienen licenciatura?": "filtro",
    "¿Cuántos empleados de Monterrey ganan menos de 20,000?": "filtro",
    "¿Cuántos empleados contratados después de 2023 trabajan en Ventas?": "filtro",
    "¿Cuántos empleados con experiencia menor a 2 años trabajan en IT?": "filtro",
    "¿Cuántos empleados de CDMX tienen maestría?": "filtro",
    "¿Cuántos empleados menores de 25 años trabajan en Marketing?": "filtro",
    "¿Cuántos empleados con licenciatura trabajan en Guadalajara?": "filtro",
    "¿Cuántos empleados en Recursos Humanos ganan más de 30,000?": "filtro",
    "¿Cuántos empleados contratados en 2022 trabajan en IT?": "filtro",
    "¿Cuántos empleados con doctorado trabajan en Finanzas?": "filtro",
    "¿Cuántos empleados mayores de 35 años trabajan en Ventas?": "filtro",
    "¿Cuántos empleados con experiencia entre 3 y 5 años trabajan en Marketing?": "filtro",
    "¿Cuántos empleados trabajan en Toluca con salario mayor a 25,000?": "filtro",
    "¿Cuántos empleados menores de 30 años trabajan en IT?": "filtro",
    "¿Cuántos empleados tienen experiencia en ventas y licenciatura?": "filtro",
    "¿Cuántos empleados trabajan en Monterrey y tienen menos de 2 años de experiencia?": "filtro",
    "¿Cuántos empleados en IT con licenciatura ganan entre 20,000 y 30,000?": "filtro",
    "¿Cuántos empleados en Finanzas tienen maestría y más de 5 años de experiencia?": "filtro",
    "¿Cuántos empleados en Ventas son menores de 30 años y tienen licenciatura?": "filtro",
    "¿Cuántos empleados en Marketing con maestría trabajan en Guadalajara?": "filtro",
    "¿Cuántos empleados en IT con doctorado fueron contratados después de 2022?": "filtro",
    "¿Cuántos empleados en Finanzas tienen licenciatura y menos de 3 años de experiencia?": "filtro",
    "¿Cuántos empleados en Ventas con maestría ganan más de 25,000?": "filtro",
    "¿Cuántos empleados en Marketing con licenciatura trabajan en Monterrey?": "filtro",
    "¿Cuántos empleados en IT tienen maestría y más de 10 años de experiencia?": "filtro",
    "¿Cuántos empleados en Finanzas con doctorado son mayores de 35 años?": "filtro",
    "¿Cuántos empleados en Ventas con licenciatura trabajan en Hermosillo?": "filtro",
    "¿Cuántos empleados en Marketing con maestría tienen menos de 2 años de antigüedad?": "filtro",
    "¿Cuántos empleados en IT con licenciatura ganan menos de 20,000?": "filtro",
    "¿Cuántos empleados en Finanzas con maestría trabajan en Monterrey?": "filtro",
    "¿Cuántos empleados en Ventas con doctorado fueron contratados en 2023?": "filtro",

    # Tipo: prediccion (20 preguntas)
    "¿Cuánto ganaría un empleado de 30 años en IT?": "prediccion",
    "¿Cuál sería el salario esperado para un empleado con 5 años de experiencia en Marketing?": "prediccion",
    "¿Qué salario debería tener un empleado nuevo en Ventas?": "prediccion",
    "¿Cuánto ganaría un empleado con doctorado en Finanzas?": "prediccion",
    "¿Cuál es el salario probable para un empleado de 25 años en Recursos Humanos?": "prediccion",
    "¿Qué salario podría ganar un empleado con 10 años de experiencia en IT?": "prediccion",
    "¿Cuál sería el sueldo esperado para un empleado en Guadalajara con licenciatura?": "prediccion",
    "¿Cuánto ganaría un empleado con licenciatura en Monterrey?": "prediccion",
    "¿Cuál es el salario esperado para un empleado menor de 30 años en Marketing?": "prediccion",
    "¿Qué salario podría tener un empleado con 3 años de experiencia en ventas?": "prediccion",
    "¿Cuál sería el sueldo estimado para un empleado nuevo en IT?": "prediccion",
    "¿Cuánto ganaría un empleado con maestría en Finanzas?": "prediccion",
    "¿Cuál es el salario probable para un empleado con experiencia media en Marketing?": "prediccion",
    "¿Qué salario tendría un empleado nuevo en Ventas con título técnico?": "prediccion",
    "¿Cuál sería el sueldo esperado para un empleado con doctorado en IT?": "prediccion",
    "¿Cuánto ganaría un empleado con experiencia en ventas y licenciatura?": "prediccion",
    "¿Cuál es el salario esperado para un empleado con menos de 1 año de experiencia en Marketing?": "prediccion",
    "¿Qué salario debería tener un empleado en Monterrey con 4 años de experiencia?": "prediccion",
    "¿Cuál es el salario probable para un empleado con experiencia en Guadalajara?": "prediccion",
    "¿Cuánto ganaría un empleado nuevo con experiencia previa en IT?": "prediccion",
}

#Banco de stop_words en español
stop_words_esp = [
    "a", "acá", "ahí", "ajena", "ajeno", "ajenas", "ajenos", "al", "algo", 
    "algún", "alguna", "alguno", "algunos", "allá", "allí", "ambos", "ante", 
    "antes", "aquel", "aquella", "aquello", "aquellos", "aquí", "arriba", 
    "así", "atrás", "aun", "aunque", "bajo", "bastante", "bien", "cabe", 
    "cada", "casi", "cierta", "cierto", "ciertas", "ciertos", "como", "con", 
    "conmigo", "conseguimos", "conseguir", "consigo", "consigue", "consiguen", 
    "consigues", "contigo", "contra", "cual", "cuales", "cualquiera", 
    "cualesquiera", "cuan", "cuando", "cuanta", "cuanto", "cuantas", "cuantos", 
    "de", "dejar", "del", "demás", "demasiada", "demasiado", "demasiadas", 
    "demasiados", "dentro", "desde", "donde", "dos", "el", "él", "ella", 
    "ello", "ellos", "empleáis", "emplean", "emplear", "empleas", "empleo", 
    "en", "encima", "entonces", "entre", "era", "eras", "eramos", "eran", 
    "eres", "es", "esa", "ese", "eso", "esas", "esos", "esta", "está", 
    "estas", "estaba", "estado", "estáis", "estamos", "están", "estar", 
    "este", "esto", "estos", "estoy", "etc", "fin", "fue", "fueron", "fui", 
    "fuimos", "bueno", "ha", "hace", "haces", "hacéis", "hacemos", "hacen", 
    "hacer", "hacia", "hago", "hasta", "incluso", "intenta", "intentas", 
    "intentáis", "intentamos", "intentan", "intentar", "intento", "ir", 
    "jamás", "junta", "junto", "juntas", "juntos", "la", "lo", "las", "los", 
    "largo", "más", "me", "menos", "mi", "mis", "mía", "mías", "mientras", 
    "mío", "míos", "misma", "mismo", "mismas", "mismos", "modo", "mucha", 
    "mucho", "muchas", "muchos", "muchísima", "muchísimo", "muchísimas", 
    "muchísimos", "muy", "nada", "ni", "ningún", "ninguna", "ninguno", 
    "ningunas", "ningunos", "no", "nos", "nosotras", "nosotros", "nuestra", 
    "nuestro", "nuestras", "nuestros", "nunca", "os", "otra", "otro", 
    "otras", "otros", "para", "parecer", "pero", "poca", "poco", "pocas", 
    "pocos", "podéis", "podemos", "poder", "podría", "podrías", "podríais", 
    "podríamos", "podrían", "por", "por qué", "porque", "primero", "puede", 
    "pueden", "puedo", "pues", "que", "qué", "querer", "quién", "quiénes", 
    "quienesquiera", "quienquiera", "quizá", "quizás", "sabe", "sabes", 
    "saben", "sabéis", "sabemos", "saber", "se", "según", "ser", "si", "sí", 
    "siempre", "siendo", "sin", "sino", "so", "sobre", "sois", "solamente", 
    "solo", "sólo", "somos", "soy", "sr", "sra", "sres", "sta", "su", "sus", 
    "suya", "suyo", "suyas", "suyos", "tal", "tales", "también", "tampoco", 
    "tan", "tanta", "tanto", "tantas", "tantos", "te", "tenéis", "tenemos", 
    "tener", "tengo", "ti", "tiempo", "tiene", "tienen", "toda", "todo", 
    "todas", "todos", "tomar", "trabaja", "trabajo", "trabajáis", "trabajamos", 
    "trabajan", "trabajar", "trabajas", "tras", "tú", "tu", "tus", "tuya", 
    "tuyo", "tuyas", "tuyos", "último", "ultimo", "un", "una", "uno", "unos", 
    "unas", "usa", "usas", "usáis", "usamos", "usan", "usar", "uso", 
    "usted", "ustedes", "va", "van", "vais", "valor", "vamos", "varias", 
    "varios", "vaya", "verdadera", "vosotras", "vosotros", "voy", "vuestra", 
    "vuestro", "vuestras", "vuestros", "y", "ya", "yo"
]

#Definición de keywords para facilitar la construcción de queries. 

keywords = {
    "departamentos": ["IT", "Finanzas", "Ventas", "Marketing"],
    "ciudades": ["Guadalajara", "Monterrey", "Hermosillo"],
    "titulos": ["Licenciatura", "maestria", "doctorado"],
    "estadistica" : ["desviacion", "promedio", ""],
    "caso":["empleado", "salario", "gana", "experiencia"]
}

casosNoValidos = {
    "gana":"salario",
    "experiencia":"experiencia_anos"
}
df = pd.DataFrame(list(preguntas.items()), columns=["Pregunta", "Tipo pregunta"])
#Separar df en grupos de train y test. 
x = df['Pregunta']
y = df['Tipo pregunta']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

#Definición del pipeline con los modelos que se emplearan, escogí el RFC por que ya lo he usado y conozco sus aplicaciones
#El TfidVectorizer convierte la información en vectores de palabras. 
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words=stop_words_esp,
        strip_accents='ascii',
        max_features=5000
    )),
    ('rfc', RandomForestClassifier())
])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
print(classification_report(y_test, y_pred))



def chatbot(pregunta: str):
    #Categoria, departamento, titulo, ciudad, tipo_estadistica. Si es None, no se extrajo la información de la pregunta. 
    info = procesar_pregunta(pregunta)
    categoria, departamento, titulo, ciudad, tipo_estadistica, caso, edad, experiencia_anios = info
    query, parametros, categoria = crear_query(categoria, departamento, titulo, ciudad, tipo_estadistica, caso, edad, experiencia_anios)
    return query, parametros, categoria


def procesar_pregunta(pregunta: str):
    categoria = pipeline.predict([pregunta])[0]
    numeros = re.findall(r'\d+', pregunta)
    numeros = [int(n) for n in numeros]
    edad = None
    experiencia_anios = None
    for numero in numeros: 
        if numero>20 and edad is None: 
            edad = numero
        if numero <= 20 and experiencia_anios is None: 
            experiencia_anios = numero
    departamento = next((d for d in keywords["departamentos"] if d.lower() in pregunta.lower()), None)
    titulo = next((d for d in keywords["titulos"] if d.lower() in pregunta.lower()), None)
    ciudad = next((d for d in keywords["ciudades"] if d.lower() in pregunta.lower()), None)
    tipo_estadistica = next((d for d in keywords["estadistica"] if d.lower() in pregunta.lower()), None)
    caso = next((d for d in keywords["caso"] if d.lower() in pregunta.lower()), None)
    #Respuesta, categoria y confianza.  
    info = [categoria, departamento, titulo, ciudad, tipo_estadistica, caso, edad ,experiencia_anios]
    return (info)

def validarCaso(caso: str):
    if caso is None: 
        return "salario"
    casosNoValidos = {
    "gana":"salario",
    "experiencia":"experiencia_anos",
    "antiguedad" : "experiencia_anos",
    "años" : "experiencia_anos",
    }
    casoValidado = casosNoValidos.get(caso.lower(), caso)
    return casoValidado
def crear_query(categoria: str, departamento: str, titulo: str, ciudad: str, tipo_estadistica: str, caso: str, edad: int, experiencia_anios: int):
    #validador de caso
    caso = validarCaso(caso) if caso else None
    condiciones = []
    parametros = []

    #Crear consulta en base al caso
    if categoria == 'filtro':
        query = 'SELECT COUNT(*) as count FROM empleados '

    elif (categoria == 'busqueda_max'):
        query = f'SELECT MAX({caso}) as max FROM empleados '

    elif (categoria == 'busqueda_min'):
        query = f'SELECT MIN({caso}) as min FROM empleados '

    elif (categoria == 'conteo'):
        query = f'SELECT COUNT(*) as count FROM empleados '
    
    elif (categoria == 'predicciones'):
        prediccion = regresion(edad, experiencia_anios, departamento, titulo)
        return prediccion

    elif (categoria == 'estadistica'):
        if tipo_estadistica is None: 
            ValueError

        elif tipo_estadistica == "promedio" or tipo_estadistica == "media":
            query = f'SELECT AVG({caso}) FROM empleados'
        elif tipo_estadistica is "mediana":
            query = query = f'''
            SELECT {caso} as mediana FROM empleados 
            ORDER BY {caso} 
            LIMIT 1 OFFSET (SELECT COUNT(*) FROM empleados)/2
            '''
    else: 
        return ("Error")

    if departamento is not None: 
        condiciones.append("departamento = ?")
        parametros.append(departamento)
            
    if titulo is not None: 
        condiciones.append("nivel_educacion = ?")
        parametros.append(titulo)

    if ciudad is not None: 
        condiciones.append("ciudad = ?")
        parametros.append(ciudad)
        
    if condiciones: 
        query += "WHERE " + "AND".join(condiciones)

    return query, parametros, categoria

        
