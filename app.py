
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PRINCIPAL
# Interfaz de Streamlit
# Selección de página
st.sidebar.image("logo.png")
pagina = st.sidebar.selectbox("Selecciona una página:", ["🏠 Home", "📋 Carga del dataset"])

#Selección Home
if pagina == "🏠 Home":
    # HOME - PRESENTACIÓN DEL PROYECTO
    # CONFIGURACIÓN
    st.set_page_config(
        page_title="Proyecto EDA Bank Marketing",
        page_icon="📊",
        layout="wide"
    )

    # ESTILOS
    st.markdown("""
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .titulo {
        text-align: center;
        color: #0E1117;
        font-size: 50px !important;
        font-weight: 900 !important;
        margin-top: -20px;
        margin-bottom: 20px;
        line-height: 1.1;
    }

    .subtitulo {
        text-align: center;
        color: #4F4F4F;
        font-size: 20px;
    }

    .card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    </style>
    """, unsafe_allow_html=True)

    # ======================================================
    # TÍTULO
    # ======================================================

    st.markdown(
        """
        <h1 class="titulo">
            📊 PROYECTO EDA - BANK MARKETING
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitulo">Análisis Exploratorio de Datos utilizando Python y Streamlit</p>',
        unsafe_allow_html=True
    )

    st.divider()

    # ======================================================
    # DESCRIPCIÓN DEL PROYECTO
    # ======================================================

    st.markdown("## 🎯 Objetivo del Proyecto")

    st.markdown("""
    <div class="card">

    El objetivo de este proyecto es desarrollar un análisis exploratorio de datos (EDA)
    sobre un dataset de campañas de marketing bancario.

    A través del uso de herramientas de análisis y visualización,
    se busca identificar patrones, distribuciones,
    relaciones entre variables y obtener insights relevantes
    para la toma de decisiones.

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # INFORMACIÓN DEL AUTOR
    # ======================================================

    st.markdown("## 👩‍💻 Datos del Autor")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        👤 Nombre:
        
        Ana Fernanda Rashel Fernández Pamucena
        """)

    with col2:
        st.info("""
        🎓 Curso:
        
        Especialización en Python
        """)

    with col3:
        st.info("""
        📅 Año:
        
        2026
        """)

    # ======================================================
    # DATASET
    # ======================================================

    st.markdown("## 📂 Descripción del Dataset")

    st.markdown("""
    <div class="card">

    El dataset utilizado corresponde a campañas de marketing bancario,
    y contiene información relacionada con clientes,
    contactos telefónicos, variables económicas
    y resultados de campañas previas.

    Entre las variables analizadas destacan:

    - Edad del cliente
    - Tipo de trabajo
    - Estado civil
    - Nivel educativo
    - Tipo de contacto
    - Duración de llamadas
    - Resultado de campañas anteriores

    El análisis permitirá comprender mejor
    el comportamiento de los clientes
    y detectar posibles patrones de conversión.

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # TECNOLOGÍAS
    # ======================================================

    st.markdown("## 🛠️ Tecnologías Utilizadas")

    tec1, tec2, tec3, tec4 = st.columns(4)

    with tec1:
        st.success("🐍 Python")

    with tec2:
        st.success("📊 Pandas")

    with tec3:
        st.success("🎨 Streamlit")

    with tec4:
        st.success("📈 Matplotlib / Seaborn")

    # ======================================================
    # FOOTER
    # ======================================================

    st.divider()

    st.caption("Proyecto desarrollado para fines académicos - EDA Bank Marketing")
    
    # Pie de página
    st.markdown("---")
    st.write("© 2026 - Proyecto Académico")

#Selección Carga del dataset
elif pagina == "📋 Carga del dataset":
    # Inicializar en sesión
    # CONFIGURACIÓN
    # ======================================================
    st.set_page_config(
        page_title="EDA Bank Marketing",
        page_icon="📊",
        layout="wide"
    )

    sns.set_style("whitegrid")

    # ESTILOS
    # ======================================================
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }

    h1 {
        color: #0E1117;
        text-align: center;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    }

    .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ======================================================
    # TÍTULO
    # ======================================================
    st.title("📊 Dashboard EDA - Bank Marketing")
    st.markdown("### Análisis Exploratorio de Datos con Streamlit + POO")
    st.divider()

    # CLASE POO
    # ======================================================
    class DataAnalyzer:

        def __init__(self, df):
            self.df = df

        def clasificar_variables(self):
            numericas = self.df.select_dtypes(include=np.number).columns.tolist()
            categoricas = self.df.select_dtypes(include=['object']).columns.tolist()
            return numericas, categoricas

        def estadisticas(self):
            return self.df.describe().T

        def valores_nulos(self):
            return self.df.isnull().sum()

        def info_general(self):
            info = pd.DataFrame({
                'Tipo de dato': self.df.dtypes,
                'Nulos': self.df.isnull().sum(),
                'Únicos': self.df.nunique()
            })
            return info

        def moda(self, columna):
            return self.df[columna].mode()[0]
        
    # SIDEBAR
    # ======================================================
    st.sidebar.title("⚙️ Panel de Control")

    mostrar_dataset = st.sidebar.checkbox("Mostrar dataset completo")
    mostrar_correlacion = st.sidebar.checkbox("Mostrar matriz de correlación")

    bins = st.sidebar.slider(
        "Número de bins",
        min_value=5,
        max_value=50,
        value=20
    )

    # CARGA DATASET
    # ======================================================
    st.header("📂 Carga del Dataset")

    archivo = st.file_uploader(
        "Sube el archivo BankMarketing.csv",
        type=['csv']
    )

    # VALIDACIÓN
    # ======================================================
    if archivo is not None:

        df = pd.read_csv(archivo, sep=';')

        st.success("✅ Dataset cargado correctamente")

        # ======================================================
        # VISTA PREVIA
        # ======================================================
        st.subheader("👀 Vista previa")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # ======================================================
        # MÉTRICAS
        # ======================================================
        filas, columnas = df.shape

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📌 Filas", filas)

        with col2:
            st.metric("📌 Columnas", columnas)

        with col3:
            st.metric("📌 Valores nulos", int(df.isnull().sum().sum()))

        if mostrar_dataset:
            st.subheader("📄 Dataset completo")
            st.dataframe(df, use_container_width=True)

        # OBJETO POO
        # ======================================================
        analyzer = DataAnalyzer(df)

        numericas, categoricas = analyzer.clasificar_variables()

        # ======================================================
        # TABS
        # ======================================================
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            "Información",
            "Variables",
            "Estadísticas",
            "Nulos",
            "Distribuciones",
            "Categóricas",
            "Num vs Cat",
            "Cat vs Cat",
            "Dinámico",
            "Insights",
            "Conclusión"
        ])

        # TAB 1
        # ======================================================
        with tab1:

            st.header("📌 Información General")

            st.dataframe(
                analyzer.info_general(),
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Tipos de datos")
                st.dataframe(df.dtypes.astype(str))

            with col2:
                st.subheader("Valores nulos")
                st.dataframe(df.isnull().sum())

        # ======================================================
        # TAB 2
        # ======================================================
        with tab2:

            st.header("📌 Clasificación de Variables")

            st.markdown("### Distribución de variables del dataset")

            col1, col2 = st.columns(2)

            # --------------------------------------------------
            # VARIABLES NUMÉRICAS
            # --------------------------------------------------
            with col1:

                st.subheader("🔢 Variables Numéricas")

                st.metric(
                    label="Cantidad",
                    value=len(numericas)
                )

                num_df = pd.DataFrame({
                    "Variables Numéricas": numericas
                })

                st.dataframe(
                    num_df,
                    use_container_width=True,
                    height=400
                )

            # --------------------------------------------------
            # VARIABLES CATEGÓRICAS
            # --------------------------------------------------
            with col2:

                st.subheader("🔤 Variables Categóricas")

                st.metric(
                    label="Cantidad",
                    value=len(categoricas)
                )

                cat_df = pd.DataFrame({
                    "Variables Categóricas": categoricas
                })

                st.dataframe(
                    cat_df,
                    use_container_width=True,
                    height=400
                )

            st.divider()

            # --------------------------------------------------
            # RESUMEN
            # --------------------------------------------------
            resumen = pd.DataFrame({
                "Tipo": ["Numéricas", "Categóricas"],
                "Cantidad": [len(numericas), len(categoricas)]
            })

            st.subheader("📊 Resumen General")

            st.dataframe(
                resumen,
                use_container_width=True
            )

        # TAB 3
        # ======================================================
        with tab3:

            st.header("📌 Estadísticas Descriptivas")

            stats = analyzer.estadisticas()

            st.dataframe(stats, use_container_width=True)

            variable = st.selectbox(
                "Seleccione variable numérica",
                numericas
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Media", round(df[variable].mean(), 2))

            with col2:
                st.metric("Mediana", round(df[variable].median(), 2))

            with col3:
                st.metric("Desv. Std", round(df[variable].std(), 2))

        # ======================================================
        # TAB 4
        # ======================================================
        with tab4:

            st.header("📌 Valores Faltantes")

            nulls = df.isnull().sum()

            st.dataframe(nulls)

            fig, ax = plt.subplots(figsize=(10,4))

            nulls.plot(kind='bar', ax=ax)

            plt.xticks(rotation=45)
            plt.title("Valores faltantes")

            st.pyplot(fig)

        # TAB 5
        # ======================================================
        with tab5:

            st.header("📌 Distribución Variables Numéricas")

            variable_num = st.selectbox(
                "Seleccione variable",
                numericas,
                key='hist'
            )

            fig, ax = plt.subplots(figsize=(10,5))

            sns.histplot(
                data=df,
                x=variable_num,
                bins=bins,
                kde=True,
                ax=ax
            )

            plt.title(f"Distribución de {variable_num}")

            st.pyplot(fig)

        # TAB 6
        # ======================================================
        with tab6:

            st.header("📌 Variables Categóricas")

            variable_cat = st.selectbox(
                "Seleccione variable categórica",
                categoricas,
                key='cat'
            )

            conteo = df[variable_cat].value_counts()

            col1, col2 = st.columns([1,2])

            with col1:
                st.dataframe(conteo)

            with col2:
                fig, ax = plt.subplots(figsize=(10,5))

                sns.countplot(
                    data=df,
                    x=variable_cat,
                    ax=ax
                )

                plt.xticks(rotation=45)
                plt.title(f"Conteo de {variable_cat}")

                st.pyplot(fig)


        # ======================================================
        # TAB 7
        # ======================================================
        with tab7:

            st.header("📌 Análisis Bivariado Numérico vs Categórico")

            num = st.selectbox(
                "Variable numérica",
                numericas,
                key='num1'
            )

            cat = st.selectbox(
                "Variable categórica",
                categoricas,
                key='cat1'
            )

            fig, ax = plt.subplots(figsize=(10,5))

            sns.boxplot(
                data=df,
                x=cat,
                y=num,
                ax=ax
            )

            plt.xticks(rotation=45)

            st.pyplot(fig)

        # TAB 8
        # ======================================================
        with tab8:

            st.header("📌 Análisis Categórico vs Categórico")

            cat_a = st.selectbox(
                "Primera variable",
                categoricas,
                key='a'
            )

            cat_b = st.selectbox(
                "Segunda variable",
                categoricas,
                key='b'
            )

            tabla = pd.crosstab(df[cat_a], df[cat_b])

            st.dataframe(tabla, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10,5))

            tabla.plot(kind='bar', stacked=True, ax=ax)

            plt.xticks(rotation=45)

            st.pyplot(fig)

        # TAB 9
        # ======================================================
        with tab9:

            st.header("📌 Análisis Dinámico")

            columnas = st.multiselect(
                "Seleccione columnas",
                df.columns.tolist(),
                default=df.columns.tolist()[:5]
            )

            if columnas:
                st.dataframe(df[columnas].head(), use_container_width=True)

            filtro = st.selectbox(
                "Seleccione filtro",
                categoricas
            )

            valor = st.selectbox(
                "Seleccione valor",
                df[filtro].unique()
            )

            filtrado = df[df[filtro] == valor]

            st.subheader("Datos filtrados")
            st.dataframe(filtrado.head(), use_container_width=True)

            st.info(f"Registros encontrados: {len(filtrado)}")

        # TAB 10
        # ======================================================
        with tab10:

            st.header("📌 Hallazgos Clave")

            col1, col2 = st.columns(2)

            # --------------------------------------------------
            # RESUMEN ESTADÍSTICO
            # --------------------------------------------------
            with col1:

                st.subheader("📊 Resumen Estadístico")

                resumen = pd.DataFrame({
                    'Media': df[numericas].mean(),
                    'Mediana': df[numericas].median(),
                    'Desv Std': df[numericas].std()
                })

                st.dataframe(
                    resumen,
                    use_container_width=True
                )

            # --------------------------------------------------
            # MODAS
            # --------------------------------------------------
            with col2:

                st.subheader("📌 Moda Variables Categóricas")

                modas = {}

                for col in categoricas:
                    modas[col] = analyzer.moda(col)

                moda_df = pd.DataFrame(
                    modas.items(),
                    columns=['Variable', 'Moda']
                )

                st.dataframe(
                    moda_df,
                    use_container_width=True
                )

            st.divider()

            # --------------------------------------------------
            # INSIGHTS
            # --------------------------------------------------
            st.subheader("💡 Principales Insights")

            st.success("✅ Hallazgos encontrados durante el EDA")

            st.write(
                "- El dataset contiene información relevante sobre campañas de marketing bancario."
            )

            st.write(
                "- Existen variables numéricas con distribuciones asimétricas."
            )

            st.write(
                "- Algunas variables categóricas presentan categorías dominantes."
            )

            st.write(
                "- Los boxplots muestran diferencias entre grupos categóricos."
            )

            st.write(
                "- El análisis dinámico permite explorar patrones específicos."
            )

            st.write(
                "- La implementación POO mejora la organización y reutilización del código."
            )

        # ======================================================
        # MATRIZ DE CORRELACIÓN
        # ======================================================
        if mostrar_correlacion:

            st.header("📌 Matriz de Correlación")

            corr = df[numericas].corr()

            fig, ax = plt.subplots(figsize=(12,6))

            sns.heatmap(
                corr,
                annot=True,
                cmap='coolwarm',
                ax=ax
            )

            plt.title("Matriz de Correlación")

            st.pyplot(fig)


        # TAB 11
        # ======================================================
        with tab11:
            st.header("📌 Conclusiones del Análisis Exploratorio de Datos")

            # ======================================================
            # CONCLUSIÓN 1
            # ======================================================

            st.success("""
            ### 1️⃣ La duración de la llamada influye significativamente en la respuesta del cliente

            El análisis entre la variable duration y la variable objetivo y
            mostró que las llamadas con mayor duración presentan
            una mayor tasa de aceptación de la campaña.

            ✅ Toma de decisión:
            La empresa puede fortalecer las estrategias de atención telefónica,
            capacitando a los asesores para mejorar la interacción y calidad
            de las llamadas comerciales.
            """)

            # ======================================================
            # CONCLUSIÓN 2
            # ======================================================

            st.success("""
            ### 2️⃣ Los clientes contactados múltiples veces muestran menor efectividad

            El análisis de la variable campaign evidenció que
            un exceso de contactos durante la campaña no incrementa
            la aceptación y puede generar rechazo por parte del cliente.

            ✅ Toma de decisión:
            Se recomienda optimizar la frecuencia de contacto,
            evitando saturar a los clientes con demasiadas llamadas.
            """)

            # ======================================================
            # CONCLUSIÓN 3
            # ======================================================

            st.success("""
            ### 3️⃣ Los clientes con ciertos niveles educativos presentan mayor respuesta positiva

            El análisis categórico entre education y y mostró diferencias
            entre grupos educativos respecto a la aceptación de la campaña.

            ✅ Toma de decisión:
            La empresa puede segmentar campañas de marketing
            según perfiles educativos para personalizar mejor la comunicación.
            """)

            # ======================================================
            # CONCLUSIÓN 4
            # ======================================================

            st.success("""
            ### 4️⃣ El tipo de contacto utilizado impacta en los resultados de la campaña

            El análisis entre contact y y evidenció diferencias
            en la efectividad según el canal de comunicación utilizado.

            ✅ Toma de decisión:
            La organización puede priorizar los canales de contacto
            con mejores resultados para optimizar recursos comerciales.
            """)

            # ======================================================
            # CONCLUSIÓN 5
            # ======================================================

            st.success("""
            ### 5️⃣ Variables económicas y demográficas muestran patrones relevantes

            El análisis exploratorio permitió identificar diferencias
            en variables como edad, empleo y situación económica
            entre los clientes que aceptaron y los que rechazaron la campaña.

            ✅ Toma de decisión:
            La empresa puede utilizar estos hallazgos para mejorar
            la segmentación comercial y dirigir campañas a perfiles
            con mayor probabilidad de respuesta positiva.
            """)