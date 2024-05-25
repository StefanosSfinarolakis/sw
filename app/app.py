import streamlit as st
from data_loader import load_data
from visualization import pca_plot, tsne_plot, eda_histogram, eda_scatter_plot
from machinelearning import classify_knn, classify_decision_tree, cluster_kmeans, cluster_gmm
from streamlit_navigation_bar import st_navbar
import pandas as pd
from info import Info_tab

selected_tab = st_navbar(["Data Loader", "2D Visualization", "Machine Learning","Info"])

st.title("Τεχνολογία Λογισμικού")

uploaded_file = st.file_uploader("Upload your data file here (CSV, Excel)")

if 'show_data' not in st.session_state:
    st.session_state.show_data = False

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if selected_tab == "Data Loader":
        if isinstance(data, str):
            st.error(data)
        else:
            st.write("Data Loaded Successfully")
            st.write("## Dataset")

            labels = data.iloc[:, -1]
            features = data.iloc[:, :-1]

            if st.button('Show Data'):
                st.session_state.show_data = not st.session_state.show_data

            if st.session_state.show_data:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Data Table")
                    st.write(features)
                with col2:
                    st.write("### Labels")
                    st.write(labels)

    elif selected_tab == "2D Visualization":
        st.header("2D Visualization")
        label_column = data.columns[-1]
        
        option = st.selectbox("Select Visualization", ["None", "PCA Plot", "t-SNE Plot", "EDA Charts"])
        
        if option == "PCA Plot":
            pca_fig = pca_plot(data, data.columns[:-1], label_column)
            st.plotly_chart(pca_fig)
        elif option == "t-SNE Plot":
            tsne_fig = tsne_plot(data, data.columns[:-1], label_column)
            st.plotly_chart(tsne_fig)
        elif option == "EDA Charts":
            st.write("## Exploratory Data Analysis")
            st.write("### Histogram")
            histogram_column = st.selectbox("Select Column for Histogram", data.columns[:-1])
            histogram_fig = eda_histogram(data, histogram_column, label_column)
            st.plotly_chart(histogram_fig)
            
            st.write("### Scatter Plot")
            x_column = st.selectbox("Select X", data.columns[:-1])
            y_column = st.selectbox("Select Y", data.columns[:-1], index=1)
            scatter_fig = eda_scatter_plot(data, x_column=x_column, y_column=y_column, label_column=label_column)
            st.plotly_chart(scatter_fig)

    elif selected_tab == "Machine Learning":
        ml_tab = st.selectbox("Select Task", ["Classification", "Clustering"])
        
        if ml_tab == "Classification":
            st.header("Classification")            
            st.subheader("K-Nearest Neighbors")
            k = st.slider("Select number of neighbors (k) for K-Nearest Neighbors", 1, 15, 3)
            accuracy_knn = classify_knn(data, data.columns[:-1], data.columns[-1], k)
            st.subheader("Decision Tree")               
            max_depth_dt = st.slider("Select maximum depth for Decision Tree", 1, 20, 3, key="max_depth_slider")
            accuracy_dt = classify_decision_tree(data, data.columns[:-1], data.columns[-1], max_depth=max_depth_dt)
            comparison_data = {
            "Model": ["K-Nearest Neighbors", "Decision Tree"],
            "Accuracy": [accuracy_knn, accuracy_dt]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.write("Comparison Table:")
            st.table(comparison_df)
            if accuracy_knn > accuracy_dt:
                st.write(f"K-Nearest Neighbors has a higher accuracy ({accuracy_knn:.2f}).")
            elif accuracy_knn < accuracy_dt:
                st.write(f"Decision Tree has a higher accuracy ({accuracy_dt:.2f}).")
            else:
                st.write("Both models have the same accuracy.")
        
        
        elif ml_tab == "Clustering":
            st.header("Clustering")            
            st.subheader("K-means")
            k = st.slider("Select number of Clusters", 2, 15, 3)
            silhouette_k = cluster_kmeans(data, data.columns[:-1], k)
            st.subheader("Gaussian Mixture")               
            n_components = st.slider("Select number of components", 2, 10, 3)
            silhouette = cluster_gmm(data, data.columns[:-1], n_components)
            comparison_data = {
            "Model": ["K-means", "Gaussian Mixture"],
            "Accuracy": [silhouette_k, silhouette]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.write("Comparison Table:")
            st.table(comparison_df)
            if silhouette_k > silhouette:
                st.write(f"K-means has a higher accuracy ({silhouette_k:.2f}).")
            elif silhouette_k < silhouette:
                st.write(f"Gaussian Mixture has a higher accuracy ({silhouette:.2f}).")
            else:
                st.write("Both models have the same accuracy.")
        
    elif selected_tab == "Info":
        Info_tab()
                  
