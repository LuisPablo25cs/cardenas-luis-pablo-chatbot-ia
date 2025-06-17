En este proyecto se presentan dos modelos de ML y uno de VC; a continuación se detallará cada uno. 

El primer modelo es un clasificador de intención entrenado con un banco de preguntas; se construyó con una pipeline utilizando un TfidfVectorizer, cuya función es volver los tokens un vector de características seguido de un random forest classifier. En conjunto lograron identificar con un 79% de accuracy el tipo de pregunta y, gracias a eso, ejecutar una query que respondiera a la pregunta, dando como resultado un chatbot. 

El segundo modelo es una regresión lineal múltiple ajustada a los datos de los empleados de la base de datos; el modelo se ajustó muy bien a los datos y eso se corrobora observando la r^2 de 0.95.

Por último, tenemos el OCR; utilizando pytesseract se pueden observar los caracteres alfanuméricos en una imagen y, gracias al uso de regex, clasificar los caracteres con el sentido que les demos en las expresiones. Este último, además, compara su resultado con la base de datos en búsqueda del empleado identificando la credencial. 