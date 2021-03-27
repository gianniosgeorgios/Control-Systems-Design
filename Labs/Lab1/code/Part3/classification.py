# Assuming detectors are in file "cv20_lab1_part2.py", replace with your filename.
import cv20_lab1_part3_utils as p3
import cv20_lab1_part2 as p2

import numpy as np

if __name__ == '__main__':
    descriptor = ["SURF", "HOG"]
    detector = ["Corner Detection", "Blobs Detection", "Blobs Detection using Box-Filters"]

    for j in range(2):
        if (j==0):
            print('####### SURF DESCRIPTOR ##########')
        else:
            print('####### HOG DESCRIPTOR ##########')
        # Here is a lambda which acts as a wrapper for detector function, e.g. harrisDetector.
        # The detector arguments are, in order: image, sigma, rho, k, threshold.
        for i in range(3):
            if (i==0):
                print('------Corner Detection Multiscale-------')
                detect_fun = lambda I: p2.harrisLaplacian(I, 2.3, 1, 0.001, 0.01, 1.85, 4)
            elif (i==1):
                print('------Blobs Detection Multiscale-------')
                detect_fun = lambda I: p2.hessianLaplace(I, 2, 0.07, 1.85, 4)
            else:
                print('------Blobs Detection with Box-Filters Multiscale-------')
                detect_fun = lambda I: p2.box_filters_multiscale(I, 2.5, 0.07, 1.78, 4)

            # You can use either of the following lines to extract features (HOG/SURF).
            if (j==0):
                desc_fun = lambda I, kp: p3.featuresSURF(I, kp)
            else:
                desc_fun = lambda I, kp: p3.featuresHOG(I, kp)

            # Extract features from the provided dataset.
            feats = p3.FeatureExtraction(detect_fun, desc_fun)

            accs = []
            for k in range(5):
                # Split into a training set and a test set.
                data_train, label_train, data_test, label_test = p3.createTrainTest(feats, k)

                # Perform Kmeans to find centroids for clusters.
                BOF_tr, BOF_ts = p3.BagOfWords(data_train, data_test)

                # Train an svm on the training set and make predictions on the test set
                acc, preds, probas = p3.svm(BOF_tr, label_train, BOF_ts, label_test)
                accs.append(acc)

            print('Mean accuracy for {} with {} descriptors: {:.3f}%'.format(detector[i], descriptor[j], 100.0*np.mean(accs)))
