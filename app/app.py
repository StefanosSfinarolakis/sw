import streamlit as st
from data_loader import load_data
from visualization import pca_plot, tsne_plot, eda_histogram, eda_scatter_plot
from machinelearning import classify_knn, classify_svm, cluster_kmeans, cluster_gm
from streamlit_navigation_bar import st_navbar
import pandas as pd
from info import Info_tab

#Define the navigation bar for the tabs
selected_tab = st_navbar(["Data Loader", "2D Visualization", "Machine Learning", "Info"])

#Display the Info tab
if selected_tab == "Info":
    st.markdown("<h1 style='text-align: center;'>Πληροφορίες</h1>", unsafe_allow_html=True)
    Info_tab()
else:
    st.markdown("<h1 style='text-align: center;'>Εφαρμογή Ανάλυσης Δεδομένων</h1>", unsafe_allow_html=True)

    #File uploader for data
    uploaded_file = st.file_uploader("Upload your data file here (CSV, Excel)")

    #Initialize session state for showing data
    if 'show_data' not in st.session_state:
        st.session_state.show_data = False

    if uploaded_file is not None:
        #Load the data from the uploaded file
        data = load_data(uploaded_file)

        #Data Loader tab
        if selected_tab == "Data Loader":
            if isinstance(data, str):
                st.error(data)
            else:
                st.write("Data Loaded Successfully")
                st.write("## Dataset")

                #Split data into features and labels
                labels = data.iloc[:, -1]
                features = data.iloc[:, :-1]

                #Button to toggle data display
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

        #2D Visualization tab
        elif selected_tab == "2D Visualization":
            st.header("2D Visualization")
            label_column = data.columns[-1]

            #Select visualization type
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
                #Exclude the selected x column from the y column options
                y_column_options = [col for col in data.columns[:-1] if col != x_column]
                y_column = st.selectbox("Select Y", y_column_options, index=0)
                scatter_fig = eda_scatter_plot(data, x_column=x_column, y_column=y_column, label_column=label_column)
                st.plotly_chart(scatter_fig)

        #Machine Learning tab
        elif selected_tab == "Machine Learning":
            ml_tab = st.selectbox("Select Task", ["Classification", "Clustering"])

            #Classification sub-tab
            if ml_tab == "Classification":
                st.header("Classification")
                st.subheader("K-Nearest Neighbors")
                k = st.slider("Select number of neighbors (k) for K-Nearest Neighbors", 1, 15, 3)
                accuracy_knn = classify_knn(data, data.columns[:-1], data.columns[-1], k)
                st.subheader("Support Vector Machine (SVM)")
                svm_kernel = st.selectbox("Select SVM kernel", ["linear", "poly", "rbf", "sigmoid"])
                accuracy_svm = classify_svm(data, data.columns[:-1], data.columns[-1], kernel=svm_kernel)
                
                #Display classification accuracy comparison
                comparison_data = {
                    "Model": ["K-Nearest Neighbors", "Support Vector Machine (SVM)"],
                    "Accuracy": [accuracy_knn, accuracy_svm]
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.write("Comparison Table:")
                st.table(comparison_df)
                
                #Display the model with higher accuracy
                if accuracy_knn > accuracy_svm:
                    st.write(f"K-Nearest Neighbors has a higher accuracy ({accuracy_knn:.4f}).")
                elif accuracy_knn < accuracy_svm:
                    st.write(f"Support Vector Machine (SVM) has a higher accuracy ({accuracy_svm:.4f}).")
                else:
                    st.write("Both models have the same accuracy.")

            #Clustering sub-tab
            elif ml_tab == "Clustering":
                st.header("Clustering")
                st.subheader("K-means")
                k = st.slider("Select number of Clusters", 2, 15, 3)
                score_k = cluster_kmeans(data, data.columns[:-1], k)
                st.subheader("Gaussian Mixture")
                n_components = st.slider("Select number of components", 2, 10, 3)
                score = cluster_gm(data, data.columns[:-1], n_components)
                
                #Display clustering accuracy comparison
                comparison_data = {
                    "Model": ["K-means", "Gaussian Mixture"],
                    "Accuracy": [score_k, score]
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.write("Comparison Table:")
                st.table(comparison_df)
                
                #Display the model with higher accuracy
                if score_k > score:
                    st.write(f"K-means has a higher accuracy ({score_k:.4f}).")
                elif score_k < score:
                    st.write(f"Gaussian Mixture has a higher accuracy ({score:.4f}).")
                else:
                    st.write("Both models have the same accuracy.")