from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
import numpy as np
import pickle
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from sklearn.decomposition import PCA

'''
print "Loading dataset."
xs_file = np.load('../data_and_scripts/train_inputs.npy')
ys = np.load('../data_and_scripts/train_outputs.npy')
# Create PCA datasets and transforms for 5-fold cross validation.
for i in xrange(0,5):
	print "PCA Fold:" + str(i+1)
	pca = PCA(n_components=500)
	train_indices = np.ones((xs_file.shape[0],), dtype=bool)
	train_indices[i*10000:(i+1)*10000] = False
	xs_train = pca.fit_transform(xs_file[train_indices])
	ys_train = ys[train_indices]
	np.save('pca_fold_'+str(i)+'_train_xs.npy', xs_train)
	np.save('pca_fold_'+str(i)+'_train_ys.npy', ys_train)
	xs_test = pca.transform(xs_file[~train_indices])
	ys_test = ys[~train_indices]
	np.save('pca_fold_'+str(i)+'_test_xs.npy', xs_test)
	np.save('pca_fold_'+str(i)+'_test_ys.npy', ys_test)
'''

def cross_validate(comps, view=False):
	for layer_size in [5,10,25]:
		for alpha in [0.01, 0.03, 0.1]:
			fold_accuracy = []
			for i in xrange(0, 3):
				if not view:
					xs = np.load('pca_fold_'+str(i)+'_train_xs.npy')[:,0:comps]
					ys = np.load('pca_fold_'+str(i)+'_train_ys.npy')
					DS = ClassificationDataSet(comps, nb_classes=10)
					for j in xrange(0, xs.shape[0]):
						DS.appendLinked(xs[j,:], ys[j])
					DS._convertToOneOfMany(bounds=[0,1])

				net = buildNetwork(comps, layer_size, 10, outclass=SoftmaxLayer)
				#net = buildNetwork(comps, layer_size, layer_size, 10, outclass=SoftmaxLayer)
				if not view:	
					trainer = BackpropTrainer(net, DS, learningrate=alpha)
					trainer.trainUntilConvergence(maxEpochs=4)
				test_xs = np.load('pca_fold_'+str(i)+'_test_xs.npy')[:,0:comps]
				test_ys = np.load('pca_fold_'+str(i)+'_test_ys.npy')
				preds = np.zeros(test_ys.shape)
				correct = 0
				for j in xrange(0, test_xs.shape[0]):
					if view:
						break
					pred_raw = net.activate(test_xs[j,:].tolist())
					pred = np.argmax(np.array(pred_raw))
					preds[j] = pred
					if pred == test_ys[j]:
						correct += 1
				if view:
					preds = np.load('long_result_%d_%d_%f_%d.npy' % (comps, layer_size, alpha, i))
					for j in xrange(0, preds.shape[0]):
						if preds[j] == test_ys[j]:
							correct += 1
				else:
					np.save('long_result_%d_%d_%f_%d.npy' % (comps, layer_size, alpha, i), preds)
				accuracy = float(correct)/test_xs.shape[0]
				fold_accuracy.append(accuracy)
			acc = np.sum(fold_accuracy)/3
			if view:
				#print "%d & %d & %f & %f\\\\" % (comps, layer_size, alpha, acc)
				#print "\hline"
				print acc,",",
			else:	
				print "Components: %d\tHidden Nodes: %d\tLearning Rate: %f Accuracy: %f" % (comps, layer_size, alpha, acc)
