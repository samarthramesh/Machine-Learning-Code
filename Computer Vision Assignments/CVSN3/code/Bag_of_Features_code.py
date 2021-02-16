import cv2
import numpy as np
import pickle
import math
from PA4_utils import load_image, load_image_gray
#import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from IPython.core.debugger import set_trace

def extractDescriptors(image_paths, vocab_size):
  sift = cv2.xfeatures2d.SIFT_create()
  dico = []
  for path in image_paths:
    im = load_image(path)
    image8bit = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    kp, des = sift.detectAndCompute(image8bit, None)
    for d in des:
      dico.append(d)
  return dico

def clusterDescriptors(descriptors, k):
  kmeans = KMeans(n_clusters = k).fit(descriptors)
  return kmeans

def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
  im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
  for i in range(image_count):
      for j in range(len(descriptor_list[i])):
        feature = descriptor_list[i][j]
        feature = feature.reshape(1, 128)
        idx = kmeans.predict(feature)
        im_features[i][idx] += 1

  return im_features

def build_vocabulary(image_paths, vocab_size):
  """
  This function will sample SIFT descriptors from the training images,
  cluster them with kmeans, and then return the cluster centers.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
        http://www.vlfeat.org/matlab/vl_dsift.html
          -  frames is a N x 2 matrix of locations, which can be thrown away
          here (but possibly used for extra credit in get_bags_of_sifts if
          you're making a "spatial pyramid").
          -  descriptors is a N x 128 matrix of SIFT features
        Note: there are step, bin size, and smoothing parameters you can
        manipulate for dsift(). We recommend debugging with the 'fast'
        parameter. This approximate version of SIFT is about 20 times faster to
        compute. Also, be sure not to use the default value of step size. It
        will be very slow and you'll see relatively little performance gain
        from extremely dense sampling. You are welcome to use your own SIFT
        feature code! It will probably be slower, though.
  -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
          http://www.vlfeat.org/matlab/vl_kmeans.html
            -  X is a N x d numpy array of sampled SIFT features, where N is
               the number of features sampled. N should be pretty large!
            -  K is the number of clusters desired (vocab_size)
               cluster_centers is a K x d matrix of cluster centers. This is
               your vocabulary.

  Args:
  -   image_paths: list of image paths.
  -   vocab_size: size of vocabulary

  Returns:
  -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
      cluster center / visual word
  """
  # Load images from the training set. To save computation time, you don't
  # necessarily need to sample from all images, although it would be better
  # to do so. You can randomly sample the descriptors from each image to save
  # memory and speed up the clustering. Or you can simply call vl_dsift with
  # a large step size here, but a smaller step size in get_bags_of_sifts.
  #
  # For each loaded image, get some SIFT features. You don't have to get as
  # many SIFT features as you will in get_bags_of_sift, because you're only
  # trying to get a representative sample here.
  #
  # Once you have tens of thousands of SIFT features from many training
  # images, cluster them with kmeans. The resulting centroids are now your
  # visual word vocabulary.

  dim = 128      # length of the SIFT descriptors that you are going to compute.
  vocab = np.zeros((vocab_size,dim))

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################
  descriptors = extractDescriptors(image_paths, vocab_size)
  kmeans = clusterDescriptors(descriptors, vocab_size)
  


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return kmeans

def get_bags_of_sifts(image_paths, vocab_filename, kmeans, vocab_size):
  """
  This feature representation is described in the handout, lecture
  materials, and Szeliski chapter 14.
  You will want to construct SIFT features here in the same way you
  did in build_vocabulary() (except for possibly changing the sampling
  rate) and then assign each local feature to its nearest cluster center
  and build a histogram indicating how many times each cluster was used.
  Don't forget to normalize the histogram, or else a larger image with more
  SIFT features will look very different from a smaller version of the same
  image.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
        descriptors is a M x 128 matrix of SIFT features
          note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster
          to compute. Also, be sure not to use the default value of step size.
          It will be very slow and you'll see relatively little performance
          gain from extremely dense sampling. You are welcome to use your own
          SIFT feature code! It will probably be slower, though.
  -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
          finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

  Args:
  -   image_paths: paths to N images
  -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.

  Returns:
  -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
  """
  # load vocabulary
  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)


  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################
  
  sift = cv2.xfeatures2d.SIFT_create()
  dlist = []
  for path in image_paths:
    im = load_image(path)
    image8bit = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    kp, des = sift.detectAndCompute(image8bit, None)
    dlist.append(des)
  
  image_count = len(image_paths)
  im_features = np.array([np.zeros(vocab_size) for i in range(image_count)])  
  for i in range(image_count):
    for j in range(len(dlist[i])):
      feature = dlist[i][j]
      feature = feature.reshape(1,128)
      idx = kmeans.predict(feature)
      im_features[i][idx] +=1
      
  scale = StandardScaler().fit(im_features)        
  im_features = scale.transform(im_features)
  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return im_features

def chisqdist(point1, point2):
  sum = 0
  for i in range(len(point1)):
    sum += (math.pow((point1[i]-point2[i]), 2))/(point1[i]+point2[i])
  return sum/2
  

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k,
    metric='euclidean',):
  """
  This function will predict the category for every test image by finding
  the training image with most similar features. Instead of 1 nearest
  neighbor, you can vote based on k nearest neighbors which will increase
  performance (although you need to pick a reasonable value for k).

  Useful functions:
  -   D = sklearn_pairwise.pairwise_distances(X, Y)
        computes the distance matrix D between all pairs of rows in X and Y.
          -  X is a N x d numpy array of d-dimensional features arranged along
          N rows
          -  Y is a M x d numpy array of d-dimensional features arranged along
          N rows
          -  D is a N x M numpy array where d(i, j) is the distance between row
          i of X and row j of Y

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating
          the ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  -   metric: (optional) metric to be used for nearest neighbor.
          Can be used to select different distance functions. The default
          metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
          well for histograms

  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################
  
  if metric == 'euclidean':
    metric = 'minkowski'
  
  knn =  KNeighborsClassifier(n_neighbors = k, metric = metric)
  knn.fit(train_image_feats, train_labels)
  test_labels = knn.predict(test_image_feats)
  
  return test_labels, knn

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################



def svm_classify(train_image_feats, train_labels, test_image_feats, lam = 1, max_iter=1000):
  """
  This function will train a linear SVM for every category (i.e. one vs all)
  and then use the learned linear classifiers to predict the category of
  every test image. Every test feature will be evaluated with all 15 SVMs
  and the most confident SVM will "win". Confidence, or distance from the
  margin, is W*X + B where '*' is the inner product or dot product and W and
  B are the learned hyperplane parameters.

  Useful functions:
  -   sklearn LinearSVC
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  -   svm.fit(X, y)
  -   set(l)

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  categories = list(set(train_labels))
  
  c = 1/lam
  svm = LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=c, max_iter=max_iter)
  svm.fit(train_image_feats, train_labels)
  test_labels = svm.predict(test_image_feats)
  
  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return test_labels, svm
