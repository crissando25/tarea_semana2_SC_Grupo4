#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[9]:


import pandas as pd

# URL cruda del archivo CSV en GitHub
url = 'https://raw.githubusercontent.com/crissando25/tarea_semana2_SC_Grupo4/main/countries.csv'

# Intentar cargar el archivo CSV con diferentes delimitadores
try:
    df = pd.read_csv(url, sep=',')  # Delimitador por defecto: coma
    print("Archivo cargado con coma como delimitador")
except pd.errors.ParserError:
    try:
        df = pd.read_csv(url, sep=';')  # Intentar con punto y coma como delimitador
        print("Archivo cargado con punto y coma como delimitador")
    except pd.errors.ParserError:
        print("No se pudo cargar el archivo CSV con los delimitadores comunes")

# Mostrar las primeras 5 filas del DataFrame si se cargó correctamente
if 'df' in locals():
    print(df.head(5))


# In[10]:


# ## Conocer información básica

# In[3]:


print('Cantidad de Filas y columnas:',df.shape)
print('Nombre columnas:',df.columns)


# In[24]:


# In[4]:


df.info()


# In[5]:


df.describe()


# ## Matriz de Correlación

# In[7]:


corr = df.set_index('alpha_3').corr(numeric_only=True)
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()


# In[27]:


# URL del archivo CSV
url = 'https://raw.githubusercontent.com/crissando25/tarea_semana2_SC_Grupo4/main/countries.csv'

# Intentar cargar el archivo CSV
try:
    df_pop = pd.read_csv(url)
    print("Archivo CSV cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el archivo CSV: {e}")

# Inspeccionar las primeras filas del DataFrame
print(df_pop.head(5))

# Mostrar información sobre el DataFrame
print(df_pop.info())

# Mostrar una descripción estadística del DataFrame
print(df_pop.describe(include='all'))


# In[28]:


### Aqui vemos la población año tras año de España

# In[71]:


df_pop_es = df_pop[df_pop["country"] == 'Spain' ]
df_pop_es.head()


# In[72]:


df_pop_es.shape


# In[29]:


# ## Visualicemos datos

# In[73]:


df_pop_es.drop(['country'],axis=1)['population'].plot(kind='bar')


# In[74]:


df_pop_ar = df_pop[(df_pop["country"] == 'Argentina')]
df_pop_ar.head()
# In[75]:


df_pop_ar.shape


# In[76]:


df_pop_ar.set_index('year').plot(kind='bar')


# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# URL del archivo CSV
url = 'https://raw.githubusercontent.com/crissando25/tarea_semana2_SC_Grupo4/main/countries.csv'

# Cargar el archivo CSV
try:
    df = pd.read_csv(url)
    print("Archivo CSV cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el archivo CSV: {e}")

# Mostrar las primeras filas del DataFrame
print(df.head(5))

# Mostrar información sobre el DataFrame
print(df.info())

# Mostrar una descripción estadística del DataFrame
print(df.describe(include='all'))


# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# URL del archivo CSV
url = 'https://raw.githubusercontent.com/crissando25/tarea_semana2_SC_Grupo4/main/countries.csv'

# Intentar cargar el archivo CSV con opciones adicionales para manejar errores de tokenización
try:
    df = pd.read_csv(url, error_bad_lines=False, warn_bad_lines=True)
    print("Archivo CSV cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el archivo CSV: {e}")

# Verificar las columnas del DataFrame
print("Columnas del DataFrame:", df.columns)

# Filtrar los países hispanohablantes si la columna 'languages' existe
if 'languages' in df.columns:
    df = df.replace(np.nan, '', regex=True)
    df_espanol = df[df['languages'].str.contains('es')]
    df_espanol['population'] = df_espanol['population'].str.replace(',', '').astype(float)
    df_espanol['area'] = df_espanol['area'].str.replace(',', '').astype(float)

    # Configurar el índice y graficar
    df_espanol.set_index('alpha_3')[['population', 'area']].plot(kind='bar', rot=15, figsize=(20, 10))
    plt.title('Población y Área de Países Hispanohablantes')
    plt.ylabel('Valor')
    plt.show()

    # Detección de Outliers
    anomalies = []

    def find_anomalies(data):
        data_std = data.std()
        data_mean = data.mean()
        anomaly_cut_off = data_std * 2
        lower_limit = data_mean - anomaly_cut_off
        upper_limit = data_mean + anomaly_cut_off
        print(f"Límite inferior: {lower_limit.iloc[0]}")
        print(f"Límite superior: {upper_limit.iloc[0]}")

        for index, row in data.iterrows():
            outlier = row
            if (outlier.iloc[0] > upper_limit.iloc[0]) or (outlier.iloc[0] < lower_limit.iloc[0]):
                anomalies.append(index)
        return anomalies

    find_anomalies(df_espanol.set_index('alpha_3')[['population']])
    print(f"Anomalías detectadas: {anomalies}")

    # Quitamos BRA y USA por ser outliers y volvemos a graficar
    df_espanol = df_espanol[~df_espanol['alpha_3'].isin(['BRA', 'USA'])]

    # Graficar nuevamente
    df_espanol.set_index('alpha_3')[['population', 'area']].plot(kind='bar', rot=65, figsize=(20, 10))
    plt.title('Población y Área de Países Hispanohablantes (sin outliers)')
    plt.ylabel('Valor')
    plt.show()

    # Graficar ordenando por tamaño de población
    df_espanol.set_index('alpha_3')[['population', 'area']].sort_values(["population"]).plot(kind='bar', rot=65, figsize=(20, 10))
    plt.title('Población y Área de Países Hispanohablantes Ordenado por Población')
    plt.ylabel('Valor')
    plt.show()

    # Visualización por área
    df_espanol.set_index('alpha_3')[['area']].sort_values(["area"]).plot(kind='bar', rot=65, figsize=(20, 10))
    plt.title('Área de Países Hispanohablantes Ordenado por Área')
    plt.ylabel('Área (km²)')
    plt.show()

    # Eliminar países con área menor a 110.000 km²
    df_2 = df_espanol.set_index('alpha_3')
    df_2 = df_2[df_2['area'] > 110000]
    print(df_2)

    # Graficar nuevamente por área
    df_2[['area']].sort_values(["area"]).plot(kind='bar', rot=65, figsize=(20, 10))
    plt.title('Área de Países Hispanohablantes con Área > 110.000 km²')
    plt.ylabel('Área (km²)')
    plt.show()
else:
    print("La columna 'languages' no existe en el DataFrame.")


# In[ ]:




