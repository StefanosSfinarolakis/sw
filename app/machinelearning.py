from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def classify_knn(data, features, label_column, k):
    #Split data into features and labels
    X = data[features]
    y = data[label_column]
    #Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #Initialize and fit the K-Nearest Neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    #Predict the labels for the test set
    y_pred = knn.predict(X_test)
    #Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def classify_svm(data, features, label_column, kernel):
    #Split data into features and labels
    X = data[features]
    y = data[label_column]
    #Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #Initialize and fit the Support Vector Machine classifier
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    #Predict the labels for the test set
    y_pred = svm.predict(X_test)
    #Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def cluster_kmeans(data, features, n_clusters):
    #Extract features from the dataset
    X = data[features]
    #Initialize and fit the KMeans clustering algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    #Calculate the score for the clustering
    score = silhouette_score(X, labels)
    return score

def cluster_gm(data, features, n_components):
    #Extract features from the dataset
    X = data[features]
    #Initialize and fit the Gaussian Mixture Model clustering algorithm
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(X)
    #Calculate the score for the clustering
    score = silhouette_score(X, labels)
    return score