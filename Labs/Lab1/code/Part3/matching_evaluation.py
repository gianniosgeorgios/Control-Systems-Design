# Assuming detectors are in file "cv20_lab1_part2.py", replace with your filename.
import cv20_lab1_part3_utils as p3
import cv20_lab1_part2 as p2

if __name__ == '__main__':

    for k in range(3):
        temp = k+1 
        print('**************Image',temp,'************************')
        for j in range(2):
            if (j==0):
                print('####### SURF DESCRIPTOR ##########')
            else:
                print('####### HOG DESCRIPTOR ##########')
            # Here is a lambda which acts as a wrapper for detector function, e.g. harrisDetector.
            # The detector arguments are, in order: image, sigma, rho, k, threshold.
            for i in range(5):
                if (i==0):
                    print('------Corner Detection One Scale-------')
                    detect_fun = lambda I: p2.harrisStephens(I, 2, 2.5, 0.05, 0.005)
                elif (i==1):
                    print('------Corner Detection Multiscale-------')
                    detect_fun = lambda I: p2.harrisLaplacian(I, 2.3, 1, 0.001, 0.01, 1.85, 4)
                elif (i==2):
                    print('------Blobs Detection One Scale-------')
                    detect_fun = lambda I: p2.hessian(I, 2, 0.12)
                elif (i==3):
                    print('------Blobs Detection Multiscale-------')
                    detect_fun = lambda I: p2.hessianLaplace(I, 2, 0.07, 1.85, 4)
                else:
                    print('------Blobs Detection with Box-Filters Multiscale-------')
                    detect_fun = lambda I: p2.box_filters_multiscale(I, 2.5, 0.07, 1.78, 4)


                # You can use either of the following lines to extract features (HOG/SURF).
                if (j==0):
                    desc_fun = lambda I, kp: p3.featuresSURF(I,kp)
                else:
                    desc_fun = lambda I, kp: p3.featuresHOG(I,kp)

                # Execute evaluation by providing the above functions as arguments
                # Returns 2 1x3 arrays containing the errors
                avg_scale_errors, avg_theta_errors = p3.matching_evaluation(detect_fun, desc_fun)
                print('Avg. Scale Error for Image ',temp,': {:.3f}'.format(avg_scale_errors[k]))
                print('Avg. Theta Error for Image ',temp,': {:.3f}'.format(avg_theta_errors[k]))
