import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

def potencia(c):
    """Calcula y devuelve el conjunto potencia del 
       conjunto c.
    """
    if len(c) == 0:
        return [[]]
    r = potencia(c[:-1])
    return r + [s + [c[-1]] for s in r]

def imprime_ordenado(c):
    """Imprime en la salida estándar todos los
       subconjuntos del conjunto c (una lista de
       listas) ordenados primero por tamaño y
       luego lexicográficamente. Cada subconjunto
       se imprime en su propia línea. Los
       elementos de los subconjuntos deben ser
       comparables entre sí, de otra forma puede
       ocurrir un TypeError.
    """
    out_list=[]
    for e in sorted(c, key=lambda s: (len(s), s)):
        out_list.append(e)
    return out_list

def combinaciones(c, n):
    """Calcula y devuelve una lista con todas las
       combinaciones posibles que se pueden hacer
       con los elementos contenidos en c tomando n
       elementos a la vez.
    """
    return [s for s in potencia(c) if len(s) == n]

def showHeatmap(df, save_fig=True, save_name="Correlation_matrix2.jpg"):
  corr = df.corr()
  print(len(corr))
  x_len=6
  y_len=(x_len/3)*2
  if len(corr) > 6:
    x_len=len(corr)
    y_len=(x_len/3)*2
  plt.figure(figsize=(x_len, y_len))

  heatmap = plt.pcolormesh(corr)
  cbar = plt.colorbar(heatmap)
  columns = df.columns
  tick_list=list(np.arange(1,len(columns),1)-0.5)
  plt.xticks(tick_list,columns,rotation=90)
  plt.yticks(tick_list,columns)
  text_pos=0.3
  for i in range(len(df.columns)-1):
    for j in range(len(df.columns)-1):
      text = plt.text(j+text_pos, i+text_pos, corr.iloc[i, j].round(2),color="red",)
  plt.title("Matriz de correlación")
  plt.ylabel("var1")
  plt.xlabel("var2")
  if save_fig:
    plt.savefig(save_name, dpi=300)
  plt.show()


def showBoxPlots(df):
  for n_col, col_name in enumerate(df.columns):
    tipo=df[df.columns[n_col]].dtype
    if tipo == "object":
      print(f"Warning: Se omite columna:[{col_name}]")
      continue
    fig,ax = plt.subplots(1,2,figsize=(10,5)) #Definiremos 1 fila y 2 columnas
    ax[0].hist(df[df.columns[n_col]],orientation='horizontal',bins=20) #Accederemos al primer Axis y pintaremos un histograma en dicho Axis
    ax[1].boxplot(df[df.columns[n_col]]) #Accederemos al segundo Axis y graficaremos un boxplot
    ax[0].title.set_text(df.columns[n_col]+" Histograma")
    ax[1].title.set_text(df.columns[n_col]+" Boxplot")
    plt.show()


def showScatterPlots(df):
  lista=imprime_ordenado(combinaciones(df.columns,2))
  for a,b in lista:
    if df[a].dtype==object or df[b].dtype==object:
      print(f"Se omite [{a}] vs [{b}]")
      continue
    print(f"\n\n")
    plt.scatter(df[a],df[b])
    plt.title(f"[{a}] vs [{b}]")
    plt.xlabel(f"[{a}]")
    plt.ylabel(f"[{b}]")
    #plt.title('')
    plt.show()
    
def getMetricasErrorRegresion(y, y_tongo, name):
  mape = np.sum(np.abs((y - y_tongo)/y)) / len(y)
  mse = mean_squared_error(y , y_tongo)
  mae = mean_absolute_error(y , y_tongo)
  r2 = r2_score(y,y_tongo)
  return pd.DataFrame([mape,mse,mae,r2], columns=[name], index=["mape", "mse","mae","r2_score"])

    
def getAllErrors(y, x, n_elements):
  errores_por_combinacion = {}
  columnas = list(x.columns)
  for n_combinaciones in range(1,n_elements+1):
    error_list = pd.DataFrame(index=["mape", "mse","mae","r2_score"])
    conjuntos=combinaciones(columnas,n_combinaciones)
    for conjunto in conjuntos:
      lr = LinearRegression()
      lr.fit(x[conjunto], y)
      y_hat = lr.predict(x[conjunto]) 
      df_tmp = getMetricasErrorRegresion(y,y_hat,str(conjunto))
      error_list[df_tmp.columns[0]]=df_tmp
    errores_por_combinacion[str(n_combinaciones)]=error_list
      
  for key in errores_por_combinacion.keys():
    errores_por_combinacion[key].columns=pd.MultiIndex.from_product([ [key], errores_por_combinacion[key].columns])  

  salida = pd.DataFrame()
  for key in errores_por_combinacion.keys():
    if salida.empty:
      salida = errores_por_combinacion[key]
    else:
      salida = pd.concat([salida, errores_por_combinacion[key]], axis=1)

  salida.columns.names=["n elementos", "conjunto"]
  return salida.T
  
  
def combina(listaA, listaB):
  salida = []
  if listaA==[]:
    return listaB
  elif listaB==[]:
    return listaA
  else:
    for j in listaB:
      salida.append(list((listaA+[j])))
    return salida

def getAllErrorsStepFrowardSelection(y, x, max_elements=0, select="mae",modelo=LinearRegression):
  if select not in ["mape", "mse","mae","r2_score"]:
    raise Exception('Error: parametro select debe tener un valor de la siguiente lista: ["mape", "mse","mae","r2_score"]')
  if max_elements > len(x.columns):
    print(f"Warning: max_elements es mayor que la lista maxima de variables a considerar. Configurando el maximo posible: {len(x.columns)}")
    max_elements=len(x.columns) 
  errores_por_combinacion = {}
  columnas = x.columns.tolist()
  if max_elements == 0:
    max_elements=len(columnas)
  elegido=[]
  for level in range(max_elements):
    error_list = pd.DataFrame(index=["mape", "mse","mae","r2_score"])
    conjuntos=combina(elegido,columnas)
    for conjunto in conjuntos:
      lr = modelo()
      if  isinstance(conjunto, str):
        conjunto=[conjunto]
      lr.fit(x.loc[:,conjunto], y)
      y_hat = lr.predict(x[conjunto])
      nombre_serie = str(conjunto)
      df_tmp = getMetricasErrorRegresion(y,y_hat,nombre_serie)
      error_list[nombre_serie] = df_tmp
    errores_por_combinacion[str(level+1)] = error_list
    selected_error_serie=error_list.loc[select]
    if select == "r2_score":
      # Buscar maximo
      selected_value = selected_error_serie.index[selected_error_serie.argmax()][2:-2]
      elegido.append(columnas[selected_error_serie.argmax()])
      del columnas[selected_error_serie.argmax()]
    else:
      # Buscar Miximo
      selected_value = selected_error_serie.index[selected_error_serie.argmin()][2:-2]
      elegido.append(columnas[selected_error_serie.argmin()])
      del columnas[selected_error_serie.argmin()]

  for key in errores_por_combinacion.keys():
    errores_por_combinacion[key].columns=pd.MultiIndex.from_product([ [key], errores_por_combinacion[key].columns])  

  salida = pd.DataFrame()
  for key in errores_por_combinacion.keys():
    if salida.empty:
      salida = errores_por_combinacion[key]
    else:
      salida = pd.concat([salida, errores_por_combinacion[key]], axis=1)

  salida.columns.names=["n elementos", "conjunto"]
  return salida.T


