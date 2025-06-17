import sqlite3

conn = sqlite3.connect('./db/empresa.db')

empleados = [
    ('Pablo', 'IT', 20000, 20, 'Hermosillo', 1, 'Licenciatura', '2025-06-16'),
    ('María', 'Marketing', 25000, 28, 'Monterrey', 3, 'Maestría', '2023-09-01'),
    ('Luis', 'Finanzas', 30000, 35, 'CDMX', 7, 'Licenciatura', '2020-01-15'),
    ('Ana', 'Ventas', 22000, 26, 'Guadalajara', 2, 'Licenciatura', '2024-04-10'),
    ('Carlos', 'IT', 28000, 32, 'Querétaro', 6, 'Maestría', '2019-11-20'),
    ('Laura', 'Finanzas', 27000, 30, 'Toluca', 5, 'Licenciatura', '2022-07-05'),
    ('David', 'Marketing', 21000, 24, 'Puebla', 2, 'Licenciatura', '2023-03-17'),
    ('Sofía', 'Ventas', 19000, 22, 'León', 1, 'Licenciatura', '2024-06-30'),
    ('Fernando', 'IT', 32000, 38, 'Tijuana', 10, 'Doctorado', '2015-05-12'),
    ('Isabel', 'Finanzas', 26000, 29, 'Mérida', 4, 'Maestría', '2021-10-18'),
    ('Jorge', 'Ventas', 18000, 23, 'Chihuahua', 1, 'Técnico', '2024-01-08'),
    ('Daniela', 'Marketing', 24000, 27, 'Cancún', 3, 'Licenciatura', '2022-08-25'),
    ('Ricardo', 'IT', 29000, 34, 'CDMX', 6, 'Licenciatura', '2020-12-03'),
    ('Patricia', 'Ventas', 20000, 25, 'Morelia', 2, 'Licenciatura', '2023-05-14'),
    ('Hugo', 'Finanzas', 31000, 36, 'Saltillo', 8, 'Maestría', '2018-09-09'),
    ('Valeria', 'Marketing', 23000, 26, 'Aguascalientes', 2, 'Licenciatura', '2024-02-19'),
    ('Andrés', 'IT', 33000, 39, 'CDMX', 12, 'Doctorado', '2010-03-01'),
    ('Lucía', 'Ventas', 21000, 24, 'Culiacán', 2, 'Doctorado', '2023-11-11'),
    ('Emilio', 'Finanzas', 27000, 31, 'Guadalajara', 5, 'Licenciatura', '2021-01-22'),
    ('Camila', 'Marketing', 25000, 28, 'CDMX', 4, 'Maestría', '2022-06-06'),
]

cursor = conn.cursor()

# Crear tabla
cursor.execute('''
    CREATE TABLE IF NOT EXISTS empleados (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL, 
        departamento TEXT NOT NULL, 
        salario INTEGER NOT NULL, 
        edad INTEGER NOT NULL, 
        ciudad TEXT NOT NULL, 
        experiencia_anos INTEGER NOT NULL,
        nivel_educacion TEXT NOT NULL, 
        fecha_ingreso DATE NOT NULL
    )
''')

#Departamentos son Finanzas, Marketing, IT, Ventas
cursor.executemany(
    '''
    INSERT INTO empleados (nombre, departamento, salario, edad, ciudad, experiencia_anos, nivel_educacion, fecha_ingreso) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?) ''', empleados
    )


conn.commit()
conn.close()
