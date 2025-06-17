import sqlite3
conn = sqlite3.connect('db/empresa.db')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM empleados")
print(cursor.fetchall())