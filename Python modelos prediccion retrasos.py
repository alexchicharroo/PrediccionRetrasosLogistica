import pandas as pd
#SE USARA SKLEARN
clientes=pd.read_csv("ClientesRaw.csv", sep=";", decimal=",")
productos=pd.read_csv("ProductosRaw.csv", sep=";", decimal=",")
pedidos=pd.read_csv("PedidosRaw.csv", sep=";", decimal=",")
print(clientes.head())
print(productos.head())
print(pedidos.head())
datos=pedidos.merge(clientes, on="id_cliente", how="left") ##tomo pedidos como referencia y le añado los datos del cliente y del producto de las otras tablas
datos=datos.merge(productos, on="id_producto", how="left")
print(datos.shape)
print(datos.head())
print(datos.isnull().sum()) #ticket medio historio tiene 5 nulls
print(datos.dtypes)
datos["id_cliente"] = datos["id_cliente"].astype(str)
datos["id_pedido"] = datos["id_pedido"].astype(str)
datos["id_producto"] = datos["id_producto"].astype(str)
datos["margen_pct"] = datos["margen_pct"].str.replace("%", "", regex=False)
datos["margen_pct"] = pd.to_numeric(datos["margen_pct"], errors="coerce")/100
print(datos.dtypes)
mediana_ticket = datos["ticket_medio_historico"].median()
datos["ticket_medio_historico"] = datos["ticket_medio_historico"].fillna(mediana_ticket)
print(datos.isnull().sum()) 

##-------- DIA 2 LIMPIO BASE Y ELIJO VARIABLES---------##
datos["fecha_pedido"] = pd.to_datetime(datos["fecha_pedido"],dayfirst=True) ##fecha en la que se pidió puede ser interesante
datos["mes_pedido"] = datos["fecha_pedido"].dt.month
datos["dia_semana"] = datos["fecha_pedido"].dt.dayofweek
y=datos["entrega_tarde"]
X = datos.drop(columns=["entrega_tarde","id_pedido","id_cliente","id_producto","fecha_pedido","fecha_entrega_prevista","cliente","producto","fecha_entrega_real"])
#se quita entrega que es la objetivo, los ids no son caraccteristicas, la fecha entrega y prevista no conviene, y cliente y producto no son caracteristicas.
print(X.shape)
print(y.shape)
print(X.columns)
##Dividimos en muestra train y muestra test los datos.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)
print(X_train.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
columnas_numericas = X.select_dtypes(include=["number"]).columns
columnas_categoricas = X.select_dtypes(include=["object", "string", "category"]).columns
print(columnas_numericas)
print(columnas_categoricas)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
pre = ColumnTransformer([("numericas", StandardScaler(), columnas_numericas),("categoricas", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas)
])
X_train_pre = pre.fit_transform(X_train)
X_test_pre = pre.transform(X_test)
print(X_train.shape)
print(X_train_pre.shape)
print(X_test.shape)
print(X_test_pre.shape)

##--------DIA 3.PRIMER MODELO: REGRESIÓN LOGISTICA--------#
from sklearn.linear_model import LogisticRegression
modelo_log = LogisticRegression(max_iter=1000)
#empezamos entrenando el modelo
modelo_log.fit(X_train_pre, y_train)
y_pred=modelo_log.predict(X_test_pre)
print(y_pred[:20])
print(y_test.iloc[:20].values)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)  #71% de aciertos este modelo
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print(matriz)
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print("Precision:", precision) #de los que predijo que llegarían tarde, el 73% llegó tarde.
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print("Recall:", recall) #de todos los que llegaron tarde, consiguió predecir el 64,62% (se le escapan bastantes).
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print("F1:", f1) #tiene un equilibrio de 68,59%, razonable pero mejorable.
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#----------DIA 4. MODELO 2: ARBOL DE DECISION--------#
from sklearn.tree import DecisionTreeClassifier
modelo_arbol = DecisionTreeClassifier(max_depth=4,random_state=42)
modelo_arbol.fit(X_train_pre, y_train)
y_pred_arbol = modelo_arbol.predict(X_test_pre)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
accuracy_arbol = accuracy_score(y_test, y_pred_arbol)
precision_arbol = precision_score(y_test, y_pred_arbol)
recall_arbol = recall_score(y_test, y_pred_arbol)
f1_arbol = f1_score(y_test, y_pred_arbol)
print("Accuracy:", accuracy_arbol) #63,6% ACIERTOS DEL MODELO 
print("Precision:", precision_arbol) #64,85% DE LOS QUE PREDIJO QUE LLEGARÍAN TARDE, LLEGARON TARDE
print("Recall:", recall_arbol) #56.46% de los que llegaron tarde, solo consiguió predecirlos
print("F1:", f1_arbol) #60.36 equilibrio razonable pero mejjorable
print(confusion_matrix(y_test, y_pred_arbol))
print(classification_report(y_test, y_pred_arbol))
print("Accuracy train:", modelo_arbol.score(X_train_pre, y_train))
print("Accuracy test:", modelo_arbol.score(X_test_pre, y_test))

##si cambias a mac_depth bajan aun mas los porcentajes y accuracy train sube mucho pero test baja, lo que puede indicar
##sobreajuste. En este caso, nos quedariamos por el momento con regresión, luego Arbol de decision4.

#--------DIA 5. MODELO 3: RANDOM FOREST---------#
from sklearn.ensemble import RandomForestClassifier
modelo_rf = RandomForestClassifier(n_estimators=200,max_depth=6,random_state=42)
modelo_rf.fit(X_train_pre, y_train)
y_pred_rf = modelo_rf.predict(X_test_pre)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
print("Accuracy:", accuracy_rf) ##66.66% aciertos de modelo
print("Precision:", precision_rf) #67.94% de los que predijo que llegarian tarde llegaron tarde
print("Recall:", recall_rf) #60.54% consiguió predecir que llegarian tarde de los que llegaron tarde
print("F1:", f1_rf) #64.03% equilibrio razonable pero mejorable
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Accuracy train:", modelo_rf.score(X_train_pre, y_train))
print("Accuracy test:", modelo_rf.score(X_test_pre, y_test))
#puede haber señal de sobreajuste, el modelo es mejor que el arbol de decisión, pero sigue siendo peor que la regresión.

#--------DIA 6. COMPARAR MODELOS--------#
resultados = pd.DataFrame({"Modelo": ["Regresion Logistica", "Arbol profundidad 4","Random Forest"],
    "Accuracy": [accuracy,accuracy_arbol,accuracy_rf],
    "Precision": [precision,precision_arbol,precision_rf],
    "Recall": [recall,recall_arbol,recall_rf],
    "F1": [f1,f1_arbol,f1_rf]})
print(resultados)
##como se ha visto, es mejor elegir reg log porque tiene mayor rendimiento en todas las métricas estudiadas!!

#ahora vamos a ver los coeficientes
nombres_variables = pre.get_feature_names_out()
print(nombres_variables)
coeficientes = modelo_log.coef_[0]
importancia_log = pd.DataFrame({"variable": nombres_variables,"coeficiente": coeficientes})
importancia_log = importancia_log.sort_values(by="coeficiente",ascending=False)
print(importancia_log) #la mas importante para llegar tarde, la distancia, luego Transnova y el coste de envío. En cuanto a categoria
#la electronica, la provincia Madrid... Luego si va a Sevilla es menos retraso, si es de deporte o si el importe es alto.
importancia_log["importancia_abs"] = importancia_log["coeficiente"].abs()
importancia_log = importancia_log.sort_values(by="importancia_abs",ascending=False)
print(importancia_log.head(15)) #muestra que las mas importantes son la distancia en cuanto a retraso, Sevilla para llegar bien
#Transnova en cuanto a retraso, el coste de envio en cunto a retraso, el deporte en cunto a llegar bien, y el importe en cuanto a llegar bien. Tambien
#destacan los dias prometidos para llegar a tiempo, si es prioridad alta suele llegar mas tarde, electronica igual y rapidex favorece el llegar a tiempo.

##LO HACEMOS CON RANDOM FOREST  PARA COMPARAR
importancias_rf = modelo_rf.feature_importances_
importancia_rf = pd.DataFrame({"variable": nombres_variables,"importancia": importancias_rf})
importancia_rf = importancia_rf.sort_values(by="importancia",ascending=False)
print(importancia_rf.head(15))
#coincidien distncia, coste, importe, los demas son distintos

#varios resultados coinciden con el análisis descriptivo de Power BI: TransNova se asocia con más retrasos, 
# mientras que Sevilla, Rapidex y la categoría Deporte aparecen asociados a mejores tiempos de entrega.

#-------DIA 7. UMBRALES Y MODELO FINAL------#
prob_retraso=modelo_log.predict_proba(X_test_pre)[:,1] #nos quedamos con la que llega tarde
print(prob_retraso[:10])
#Con el umbral normal 0.50
y_pred05=(prob_retraso>=0.5).astype(int)
prec05=precision_score(y_test, y_pred05)#0.73
recall05=recall_score(y_test, y_pred05) #0.646
f105=f1_score(y_test, y_pred05) #0.68
print(confusion_matrix(y_test, y_pred05))

#Con el umbral 0.45
y_pred045=(prob_retraso>=0.45).astype(int)
prec045=precision_score(y_test, y_pred045) #0.6987
recall045=recall_score(y_test, y_pred045) #0.7415
f1045= f1_score(y_test, y_pred045)#0.719
print(confusion_matrix(y_test, y_pred045))

#Con el umbral 0.40
y_pred04=(prob_retraso>=0.4).astype(int)
prec04=precision_score(y_test, y_pred04) #0.6918
recall04=recall_score(y_test, y_pred04)#0.809
f104=f1_score(y_test, y_pred04) #0.746
print(confusion_matrix(y_test, y_pred04))

resultadosumbral = pd.DataFrame({"Umbral": ["0.50", "0.45","0.40"],
    "Precision": [prec05,prec045,prec04],
    "Recall": [recall05,recall045,recall04],
    "F1": [f105,f1045,f104]})
print(resultadosumbral) 
#el mas interesante puede ser 0.4 ya que baja la precision pero sigue siendo de 0.69, y el recall
#que es lo maz importantes, ya que son las entregas tardes que no las captabamos, sube a que captemos el 80% del 64% anterior
#ademas el f1 que es el ewuilibrio sube tambien a un 74,6%