import streamlit as st
from data_loader import load_data
from visualization import pca_plot, tsne_plot, eda_histogram, eda_scatter_plot
from machinelearning import classify_knn, classify_decision_tree, cluster_kmeans, cluster_gmm
from streamlit_navigation_bar import st_navbar

selected_tab = st_navbar(["Data Loader", "2D Visualization", "Machine Learning"])

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
            classifier = st.selectbox("Select Classifier", ["K-Nearest Neighbors", "Decision Tree"])
            
            if classifier == "K-Nearest Neighbors":
                k = st.slider("Select number of neighbors (k)", 1, 15, 3)
                accuracy = classify_knn(data, data.columns[:-1], data.columns[-1], k)
                st.write(f"Accuracy of K-Nearest Neighbors: {accuracy}")
            elif classifier == "Decision Tree":
                max_depth = st.slider("Select max depth", 1, 15, 3)
                accuracy = classify_decision_tree(data, data.columns[:-1], data.columns[-1], max_depth)
                st.write(f"Accuracy of Decision Tree: {accuracy}")
        
        elif ml_tab == "Clustering":
            st.header("Clustering")
            clusterer = st.selectbox("Select Clustering Algorithm", ["K-Means", "Gaussian Mixture"])
            
            if clusterer == "K-Means":
                n_clusters = st.slider("Select number of clusters", 2, 10, 3)
                silhouette = cluster_kmeans(data, data.columns[:-1], n_clusters)
                st.write(f"Silhouette Score of K-Means: {silhouette}")
            elif clusterer == "Gaussian Mixture":
                n_components = st.slider("Select number of components", 2, 10, 3)
                silhouette = cluster_gmm(data, data.columns[:-1], n_components)
                st.write(f"Silhouette Score of Gaussian Mixture: {silhouette}")
