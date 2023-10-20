import streamlit as st
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Desative o aviso PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Defina as informações de conexão ao banco de dados PostgreSQL
db_host = "dpg-ckonrk41tcps73b8raj0-a.oregon-postgres.render.com"
db_port = 5432
db_name = "dashboarddatabase"
db_user = "dashboardusa"
db_password = "age085yQwL1W3ZXs2pJS1Tk3QLKxr4LL"

# Tente conectar ao banco de dados
try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password,
        sslmode="require",
    )


    cursor = conn.cursor()

    # Execute uma consulta SQL na tabela do banco de dados
    cursor.execute("SELECT * FROM public.water_potability")  # Certifique-se de que a tabela está no esquema "public"

    dados = cursor.fetchall()

    # Exiba os dados em uma tabela Streamlit
    st.header('Tabela de Dados do Banco de Dados')
    df = pd.DataFrame(dados, columns=[desc[0] for desc in cursor.description])
    st.dataframe(df)

    conn.close()

except Exception as e:
    st.error(f"Erro na conexão ao banco de dados: {e}")

# Tratamento de Dados
st.subheader("Tratamento de Dados")
st.write("O conjunto de dados passou por um processo de tratamento para lidar com valores ausentes e normalizar os recursos. O tratamento de dados incluiu as seguintes etapas:")

st.write("1. Preenchimento de Valores Ausentes:")
st.write("   - Os valores ausentes foram preenchidos utilizando a estratégia da média para as variáveis numéricas.")

st.write("2. Normalização de Recursos:")
st.write("   - Os recursos foram padronizados para garantir que todos tenham a mesma escala, o que é importante para algoritmos de aprendizado de máquina baseados em distância, como o K-Nearest Neighbors (KNN).")

st.write("3. Divisão do Conjunto de Dados:")
st.write("   - O conjunto de dados foi dividido em um conjunto de treinamento e um conjunto de teste para avaliação de modelos.")



# Treinamento de modelos e exibição de precisões
if st.button("Treinar Modelos"):
    X = df.drop(['potability'], axis=1)  # Use 'potability' em letras minúsculas
    y = df['potability']

    # Preenchimento de valores ausentes com o SimpleImputer (substituindo NaN pela média)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Padronização dos dados após o tratamento dos valores ausentes
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=35)
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_leaf=30)
    rf = RandomForestClassifier(min_samples_leaf=2, n_estimators=500)
    ada = AdaBoostClassifier(learning_rate=0.2, n_estimators=50)

    knn_model = knn.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred)

    dt_model = dt.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred)

    rf_model = rf.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred)

    ada_model = ada.fit(X_train, y_train)
    y_pred = ada_model.predict(X_test)
    accuracy_ada = accuracy_score(y_test, y_pred)
    
    st.subheader("A precisão dos modelos é:")
    st.write("A precisão do modelo knn é:", accuracy_knn)
    st.write("A precisão do modelo Decision Tree é:", accuracy_dt)
    st.write("A precisão do modelo Random Forest é:", accuracy_rf)
    st.write("A precisão do modelo Ada Boost é:", accuracy_ada)


# Adicione widgets no Streamlit para análise e exibição de gráficos
if st.checkbox("Mostrar Gráfico de Correlação"):
    plt.figure(figsize=(15, 9))
    sns.heatmap(df.corr(), annot=True)
    st.pyplot()

if st.checkbox("Mostrar Estatísticas"):
    st.write(df.info())
    st.write(df.isnull().sum())

# Gráfico de Pizza da Porcentagem de Portabilidade
if st.checkbox("Mostrar Gráfico de Pizza da Porcentagem de Portabilidade"):
    st.subheader("Gráfico de Pizza da Porcentagem de Portabilidade")
    count = df['potability'].value_counts()
    labels = ['Não Potável', 'Potável']
    values = count.values
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['#FF6347', '#1E90FF'])
    st.pyplot(fig)
