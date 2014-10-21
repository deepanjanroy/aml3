import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import os

def get_sift_features(image_file_paths, k):
	''' Given a list of file paths of which to generate sift features, 
	return a numpy array where each line is the sift features for the 
	corresponding input file. 
	Algorithm:
	1) Detect all sift features in every image (vocabulary).
	2) Perform k-means clustering to clusters sift features with their closest neighbours.
	3) Perform tf-idf encoding on each sift feature in the input vector.
	4) Return the final feature set.
	'''
	vocab = []		# This is a list of every SIFT descriptor.
	raw_corpus = [] # 1 row for each image, which contains a list of all its SIFT descriptors.
	sift = cv2.SIFT()
	# Build the vocab and raw corpus.
	for f in image_file_paths:
	    if os.path.isfile(f):
	    	img = cv2.imread(f, cv2.CV_LOAD_IMAGE_GRAYSCALE)    	
	    	_, desc = sift.detectAndCompute(img, None)
	    	if desc != None:
				img_features = []
				for row in desc:
					vocab.append(row.tolist())
					img_features.append(row.tolist())
				raw_corpus.append(img_features)

	# Perform clustering with k clusters. This will probably need tuning.
	cluster = MiniBatchKMeans(k, n_init=1)
	cluster.fit(vocab)
	
	# Now we build the clustered corpus where each entry is a string containing the cluster ids for each sift-feature.
	corpus = []
	for entry in raw_corpus:
		corpus.append(' '.join([str(x) for x in cluster.predict(entry)]))
	
	# Build the Tfidf vectorizer for sift-clusters and return the transformed feature vector.
	vectorizer = TfidfVectorizer(min_df=1, token_pattern=u'\\w{1,10}\\b')
	xs = vectorizer.fit_transform(corpus)
	return xs
