import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

conn = sqlite3.connect("./db/empresa.db")
df = pd.read_sql("SELECT * FROM empleados", conn)

dummies = pd.get_dummies(df['departamento'])
df = pd.concat([df, dummies], axis = 1)
df = df.drop(columns=['departamento'])

dummies = pd.get_dummies(df['nivel_educacion'])
df = pd.concat([df, dummies], axis = 1)
df = df.drop(columns=['nivel_educacion'])

print(df.columns)

x = df[["edad", "experiencia_anos", "Doctorado", "Licenciatura", "Maestría", "Técnico" ,"Finanzas", "IT", "Marketing", "Ventas"]].values
y = df['salario'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model_regression = LinearRegression()
model_regression.fit(x_train, y_train)

x_labels = ["Edad", "Experiencia_anos", "Doctorado", "Licenciatura", "Maestría", "Técnico" ,"Finanzas", "IT", "Marketing", "Ventas"]
c_label = ['Coeficientes']
coeff_df = pd.DataFrame(model_regression.coef_, x_labels, c_label)
coeff_df

y_pred = model_regression.predict(x_test)

residuals = pd.DataFrame({'real': y_test, 'Predicción' : y_pred, 'Residual' : y_test - y_pred})
residuals = residuals.sample(n=6)
residuals = residuals.sort_values(by='real')
residuals

rsqr = (r2_score(y_test, y_pred))

conn.close()

def regresion(edad: int, experiencia: int, departamento: str, nivel_educacion : str):
    if edad == None: 
        edad = 25
    if experiencia == None: 
        experiencia == 5
    IT = Finanzas = Marketing = Ventas = False
    if departamento == "IT":
        IT = True
    elif departamento == "Finanzas":
        Finanzas = True
    elif departamento == "Marketing":
        Marketing = True
    elif departamento == "Ventas":
        Ventas = True
    else: 
        print("Error aqui")
        return "Error"
    Doctorado = Maestria = Licenciatura = Tecnico = False
    if nivel_educacion == "Doctorado":
        Doctorado = True
    elif nivel_educacion == "Maestría" or nivel_educacion == "Maestria":
        Maestria = True
    elif nivel_educacion == "Licenciatura":
        Licenciatura = True
    elif nivel_educacion == "Técnico" or nivel_educacion == "Tecnico":
        Tecnico = True
    else: 
        print("Error aca")
        return "Error"
    
    data = [edad, experiencia, Doctorado, Licenciatura, Maestria, Tecnico, Finanzas, IT, Marketing, Ventas]
    print(data)
    pred = model_regression.predict([data])
    return pred, rsqr