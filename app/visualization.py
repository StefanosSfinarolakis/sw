import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca_plot(data, features, label_column):
    #Perform PCA to reduce data to 2 dimensions
    pca = PCA(n_components=2)
    components = pca.fit_transform(data[features])
    #Create a DataFrame with the PCA components and labels
    df_pca = pd.DataFrame(data=components, columns=['Features', 'Samples'])
    df_pca['Label'] = data[label_column]
    #Create a scatter plot for the PCA results
    fig = px.scatter(df_pca, x='Features', y='Samples', color='Label', title='PCA Plot')
    return fig

def tsne_plot(data, features, label_column, random_state=None):
    #Perform t-SNE to reduce data to 2 dimensions
    tsne = TSNE(n_components=2, random_state=random_state)
    components = tsne.fit_transform(data[features])
    #Create a DataFrame with the t-SNE components and labels
    df_tsne = pd.DataFrame(data=components, columns=['Features', 'Samples'])
    df_tsne['Label'] = data[label_column]
    #Create a scatter plot for the t-SNE results
    fig = px.scatter(df_tsne, x='Features', y='Samples', color='Label', title='t-SNE Plot')    
    return fig

def eda_histogram(data, column, label_column):
    #Create a histogram for the selected column and color by label column
    fig = px.histogram(data, x=column, color=label_column, title=f'Histogram of {column}')
    return fig

def eda_scatter_plot(data, x_column, y_column, label_column):
    #Create a scatter plot for the selected x and y columns and color by label column
    fig = px.scatter(data, x=x_column, y=y_column, color=label_column, title=f'Scatter plot of {x_column} vs {y_column}', opacity=0.7)
    return fig