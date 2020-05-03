# In[0]
import numpy as np
import matplotlib.pyplot as plt
import cv2

class kmeans:
    def __init__(self, use_kmeans_plus = False, number_of_components = 5, number_of_iterations = 25):
        self.m = number_of_components
        self.iterations = number_of_iterations
        self.use_kmeans_plus = use_kmeans_plus
        self.centroid_labels = np.arange(self.m)

    def rand_initialise(self):
        self.Mu = self.X[np.random.randint(self.X.shape[0], size = self.m)]

    def assign_to_closest_centroid(self, observation_index):
        """
        1. Fetches an observation from the data using 'observation_index'.
        2. Makes as many copies of the observation as the number of clusters. 
        3. Gets the label of the cluster closest to the observation and 
            stores it in the 'Labels' array. 
        """
        xx = np.tile(self.X[observation_index], (self.d, 1))
        self.Labels[observation_index] = np.argmin(np.linalg.norm(xx - self.Mu, axis = 1))

    def update_centroid_position(self, centroid_label):
        """
        1. Fetches all observations which are assigned to 'centroid_label'. 
        2. Calculates the mean of these observations and assigns the value 
            to the corresponding centroid value. 
        """
        X_centroid = self.X[np.where(self.Labels == centroid_label)[0]]
        self.Mu[centroid_label] = np.mean(X_centroid, axis = 0)

    def get_loss(self):
        """
        1. Calculates the sum of normed distances between observations and the
            centroids that they are assigned. 
        """
        return np.linalg.norm(self.X - self.Mu[self.Labels])
    
    def fit(self, data):
        self.X = data
        self.d = data.shape[1]
        self.Labels = np.zeros(self.X.shape[0]).astype(np.uint8)
        self.observation_indeces = np.arange(self.X.shape[0])
        e_step = np.vectorize(self.assign_to_closest_centroid)
        m_step = np.vectorize(self.update_centroid_position)
        self.rand_initialise()
        losses = []
        current_loss = self.get_loss()

        for iteration in range(self.iterations):
            e_step(self.observation_indeces)
            m_step(self.centroid_labels)
            new_loss = self.get_loss()
            print("at iteration_step : " + str(iteration) + " loss = " + str(new_loss))
            losses.append(new_loss)
            if current_loss - new_loss < 0.01:
                print("converged")
            current_loss = new_loss
           
        return losses, self.Mu, self.Labels
            



