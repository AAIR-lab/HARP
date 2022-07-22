import os, pdb, pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

bounds = [[-2.5, -2.5], [2.5, 2.5]]
ppwu = 224 / (bounds[1][0] - bounds[0][0]) # number of pixels per unit in the robot world, images are all 224x224
pixelwidth = 1 / ppwu

def pixelBounds(pixel, pixelwidth=pixelwidth, bounds=bounds):
    '''
    obtain pixel bounds in terms of world coordinates
    '''
    pixminx = bounds[0][0] + (pixel[1] * pixelwidth)
    pixminy = bounds[1][1] - ((pixel[0] + 1) * pixelwidth)
    pixmaxx = bounds[0][0] + ((pixel[1] + 1) * pixelwidth)
    pixmaxy = bounds[1][1] - (pixel[0] * pixelwidth)
    b = [(pixminx, pixminy), (pixmaxx, pixmaxy)]

    return b


def dofValues(pixel_value):
    # regression predictions in [0,1]
    # denorm_d1 = (pixel_value[0] * (2 * np.pi)) + (-np.pi)
    # denorm_d2 = (pixel_value[1] * np.pi) + (-np.pi/2)
    
    # regression predictions in [-1,1]
    # denorm_d1 = ((pixel_value[0]+1) * (np.pi)) + (-np.pi)
    # denorm_d2 = ((pixel_value[1]+1) * (np.pi/2)) + (-np.pi/2)

    random_sample = 2*np.pi * np.random.random() + (-np.pi)
    # classification labels
    d1_bin_value = [     np.pi, -3*np.pi/4,   -np.pi/2,   -np.pi/4,        0,   np.pi/4,   np.pi/2, 3*np.pi/4,     np.pi,  random_sample]
    denorm_d1 = d1_bin_value[int(pixel_value[0])]

    random_sample = np.pi * np.random.random() + (-np.pi/2)
    d2_bin_value = [   -np.pi/2,  -3*np.pi/8,    -np.pi/4,    -np.pi/8,         0,    np.pi/8,    np.pi/4,  3*np.pi/8,    np.pi/2,  random_sample]
    denorm_d2 = d2_bin_value[int(pixel_value[1])]

    return [denorm_d1, denorm_d2]

def generate_samples(inp, pred, envnum):

    samplebounds = []      
    for width in xrange(pred.shape[0]):
        for length in xrange(pred.shape[1]):
            pixel = pred[width, length, 0]
            envpixel = inp[width, length, 0]

            # correct green pixels that lie in obstacle areas
            if envpixel == 0 and pixel != 0:
                pred[width, length, 0] = 0
                pixel = 0

            if pixel == 1.0:
                pbounds = pixelBounds((width, length))
                dof_values = dofValues(pred[width, length, 1:])
                # minx, miny, maxx, maxy, [dof1, dof2]
                samplebounds.append([pbounds[0][0], pbounds[0][1], pbounds[1][0], pbounds[1][1], dof_values])
    
    print("Total samples: {total_samples}".format(total_samples=len(samplebounds)))
    
    if not os.path.isdir(os.path.join('results', 'env'+envnum)):
        os.makedirs(os.path.join('results', 'env'+envnum))
    
    samples_file = open(os.path.join('results', 'env'+envnum , 'samples.pkl'), 'wb')
    pickle.dump(samplebounds, samples_file, protocol=pickle.HIGHEST_PROTOCOL)

    goal_loc = list(inp[0,0,3:])
    # data in [0,1]
    goal_loc[0] = goal_loc[0]*5 + (-2.5)
    goal_loc[1] = goal_loc[1]*5 + (-2.5)
    goal_loc[2] = (goal_loc[2] * (2 * np.pi)) + (-np.pi)
    goal_loc[3] = (goal_loc[3] * np.pi) + (-np.pi/2)

    # data in [-1,1]
    # goal_loc[0] = ((goal_loc[0]+1) * (2.5)) + (-2.5)
    # goal_loc[1] = ((goal_loc[1]+1) * (2.5)) + (-2.5)
    # goal_loc[2] = ((goal_loc[2]+1) * (np.pi)) + (-np.pi)
    # goal_loc[3] = ((goal_loc[3]+1) * (np.pi/2)) + (-np.pi/2)
    print("Goal: ", goal_loc)
    goal_file = open(os.path.join('results', 'env'+envnum , 'goal.pkl'), 'wb')
    pickle.dump(goal_loc, goal_file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    model_name = 'unet-v26'
    dataPath = os.path.join('segnet.tf-master','src','results')
    samplefiles = sorted(os.listdir(dataPath))
    for sample in samplefiles:
        if sample.endswith('.npy') and 'logits' not in sample:
            envnum = sample[:-4]
            print("processing {sample}".format(sample=sample))
            # load env image to compare with label pixel by pixel
            inp = np.load(os.path.join('datatest_cls', sample))
            # lbl = np.load(os.path.join('segnet.tf-master','src','input', 'raw', 'test-labels', envnum + '.npy'))
            pred = np.load(os.path.join('segnet.tf-master','src','results', sample))
            pred = pred.reshape((224,224,3))
            logits = np.load(os.path.join('segnet.tf-master','src', 'results', envnum + '_logits.npy'))
        
            # Round off raw predictions with sigmoid output, data in [0,1]
            # pred[:,:,0] = np.round(pred[:,:,0])
            
            # Round off raw predictions with tanh output, data in [-1,1]
            # arr2 = pred[:,:,0]
            # mask1 = (arr2[:,:] > 0) 
            # arr2[mask1] = 1
            # mask2 = (arr2[:,:] <= 0)
            # arr2[mask2] = 0
            # pred[:,:,0] = arr2

            # apply activations on classification logits
            # activations_cr = sigmoid(logits[:,:,0])
            # prediction_cr = activations_cr.copy()
            # mask = (activations_cr >= 0.5)
            # prediction_cr[mask] = 1
            # prediction_cr[~mask] = 0
            
            # activations_dof1 = softmax(logits[:,:,1:10], axis=2)
            # predictions_dof1 = tf.math.argmax(activations_dof1, axis=-1)
            
            # activations_dof2 = softmax(logits[:,:,10:], axis=2)
            # predictions_dof2 = tf.math.argmax(activations_dof2, axis=-1)

            # Plot predictions
            f, axarr = plt.subplots(1,4)
            axarr[0].imshow(inp[:,:,0], cmap='gray')
            axarr[1].imshow(pred[:,:,0], cmap='gray')
            axarr[2].imshow(pred[:,:,1], cmap='gray')
            axarr[3].imshow(pred[:,:,2], cmap='gray')
            plt.show()

            generate_samples(inp, pred, envnum)