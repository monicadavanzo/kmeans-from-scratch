import numpy as np
import matplotlib.pyplot as plt
import math

class KMeans:
  def __init__(self, n_clusters, init = 'k-means++', n_init = 'auto', max_iter = 300, tol = 1e-4, random_state = None):
    self.n_clusters = n_clusters
    self.init = init # how the initial centroids are chosen. 
    self.n_init = n_init # the number of times the algorithm will be run with different centroid seeds. 
    self.max_iter = max_iter # max number of iterations for the algorithm to converge
    self.tol = tol
    self.random_state = random_state
    self.centroids = [] 
    self.best_centroids = []
    self.clusters = []
    self.best_labels = []
    self.best_inertia = None

  def _init_centroids(self,X): 
    if self.init == 'random':
      rand_idx = np.random.choice(X.shape[0], self.n_clusters, replace = False)
      cent = X[rand_idx]
      self.centroids.append(cent)
    elif self.init == 'k-means++':
      rand = np.random.choice(X.shape[0],1)
      centers = []
      centers.append(X[rand])
      for i in range(1,self.n_clusters):
        dist = np.array([np.sqrt(np.sum((X - centers[j])**2, axis = 1)) for j in range(i)])
        nearest_dist = np.min(dist, axis = 0)
        weight = (nearest_dist**2)/np.sum(nearest_dist**2)
        next_cent_idx = np.random.choice(X.shape[0],1,p = weight)
        centers.append(X[next_cent_idx])
      centers = np.array(centers).flatten().reshape(self.n_clusters, X.shape[1])
      self.centroids.append(centers)

  def _assign_clusters(self,X,centroids):
    dist = np.array([np.sqrt(np.sum((X-centroids[i])**2, axis = 1)) for i in range(self.n_clusters)])
    assignment = np.argmin(dist,axis=0)
    self.clusters.append(assignment)
    #return assignment
  def _update_centroids(self, X, assignments):
    means = np.array([np.mean(X[assignments==c], axis = 0) for c in range(self.n_clusters)])
    self.centroids.append(means)
    #return means

  def inertia(self,X,clusters,centroids):
    inertia = np.sum([np.sum(np.sum((X[clusters == i] -centroids[i])**2,axis = 1)) for i in range(self.n_clusters)])
    return inertia
 
  def fit(self,X):
    if self.random_state is not None:
      np.random.seed(self.random_state)
    self.best_inertia = 1e16
    if self.n_init == 'auto':
      if self.init == 'k-means++':
        n_runs = 1
      else:
        n_runs = 10
    else:
      n_runs = self.n_init
    for j in range(n_runs):
      # initialize centroids
      self._init_centroids(X) # i = 0
      self._assign_clusters(X,self.centroids[0]) # i = 0
      self._update_centroids(X,self.clusters[0]) # i = 1
      self._assign_clusters(X,self.centroids[1])
      i = 1
      iter = 0
      while True:
        self._update_centroids(X,self.clusters[i])
        i+=1
        self._assign_clusters(X,self.centroids[i])
        iter += 1
        if iter >= self.max_iter:
          #self.centroids = np.array(self.centroids)
          #self.clusters = np.array(self.clusters)
          break
        if np.sqrt(np.sum((self.centroids[i] - self.centroids[i-1])**2)) < self.tol:
          #self.centroids = np.array(self.centroids)
          #self.clusters = np.array(self.clusters)
          break
      #calculate inertia
      current_inertia = self.inertia(X,self.clusters[-1], self.centroids[-1])
      #compare to current best, if better-replace
      if current_inertia < self.best_inertia:
        self.best_inertia = current_inertia
        #print(self.best_inertia)
        self.best_labels = self.clusters
        self.best_centroids = self.centroids
      # clear current run
      self.clusters = []
      self.centroids = []

  def get_labels(self):
    return self.best_labels[-1].copy()

  def get_centers(self):
    return self.best_centroids[-1].copy()
  
  def get_inertia(self):
    return np.round(self.best_inertia.copy(),2)

  def plot_process(self,X):
    #m = self.best_centroids.shape[0] # number of iterations
    m = len(self.best_centroids)
    # choose grid size (square-ish layout)
    cols = math.ceil(math.sqrt(m)) # math.ceil rounds up
    rows = math.ceil(m / cols)
    plt.figure(figsize=(4*cols, 3*rows))
    for i in range(m):
      plt.subplot(rows,cols,i+1)
      plt.title(f"Iteration {i}")
      plt.scatter(X[:,0], X[:,1],c=self.best_labels[i], cmap="winter", s=50)
      plt.scatter(self.best_centroids[i][:,0], self.best_centroids[i][:,1],c = range(self.n_clusters), cmap = 'winter', marker = 'P', s = 150, linewidths=3)
    plt.tight_layout()
    plt.show()