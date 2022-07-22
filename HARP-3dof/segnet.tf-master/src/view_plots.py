import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def plot_predictions(model_name, show_ground_truth=False):

    dataPath = os.path.join('results', model_name)
    samplefiles = glob.glob(dataPath+'/*[0-9].npy')
    total_columns = 7 if show_ground_truth else 4
    f, axarr = plt.subplots(len(samplefiles), total_columns, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(20,20))
    for i, samplefile in enumerate(samplefiles):
        sample = os.path.split(samplefile)[-1]
        print("processing {sample}".format(sample=sample))
        envnum = sample[:-4]

        # load env image to compare with label pixel by pixel
        inp = np.load(os.path.join('input', 'raw', 'test', envnum + '.npy'))
        if show_ground_truth:
            lbl =  np.load(os.path.join('input', 'raw', 'test-labels', envnum + '.npy'))
        pred = np.load(os.path.join('results', model_name, envnum + '.npy'))
        logits = np.load(os.path.join('results', model_name, envnum + '_logits.npy'))
    
        pred = pred.reshape((1,224,224,3))
        # pred[0,:,:,0] = np.round(pred[0,:,:,0])

        goal_loc = inp[0,0,3:]
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
        print(goal_loc)
            
        pw = 5./224.
        xloc_pixel = int((goal_loc[0] + 2.5)/pw)
        yloc_pixel = 224 - int((goal_loc[1] + 2.5)/pw)

        # ENV
        env = Image.fromarray(inp[:,:,0]).convert("RGB")
        for width in xrange(env.size[0]):
            for length in xrange(env.size[1]):
                pixel = env.getpixel((width, length))
                if width >= xloc_pixel - 5 and width <= xloc_pixel + 5 and \
                    length >= yloc_pixel - 5 and length <= yloc_pixel + 5:
                    env.putpixel((width, length), (0, 255, 0))
                else:
                    if pixel[0] == 1:
                        env.putpixel((width, length), (255, 255, 255))
                    else:
                        env.putpixel((width, length), (pixel[0], pixel[1], pixel[2]))
        axarr[i][0].imshow(np.array(env))
        axarr[i][0].axis("tight")
        axarr[i][0].axis("image") 
        axarr[i][0].axis('off')

        # Prediction + Env
        for j in range(3):
            inp_ = (inp[:,:,0]*255).astype(np.uint8)
            foreground = Image.fromarray(inp_, 'L').convert("RGBA")
            # pred = np.array(results['prediction_cr'][j])
            pred = pred.reshape((224,224,3))
            pred_ = (pred[:,:,j]*255).astype(np.uint8)
            if j == 1:
                mask = (pred[:,:,j] == 8)
                pred[mask] = 0
                pred_ = ((pred[:,:,j]/9.0)*255).astype(np.uint8)
            if j == 2:
                mask = (pred[:,:,j] == 9)
                pred[mask] = 0
                pred_ = ((pred[:,:,j]/10.0)*255).astype(np.uint8)
            background = Image.fromarray(pred_, 'L').convert("RGBA")
            for width in xrange(foreground.size[0]):
                for length in xrange(foreground.size[1]):
                    pixel = foreground.getpixel((width, length))
                    if width >= xloc_pixel - 5 and width <= xloc_pixel + 5 and \
                        length >= yloc_pixel - 5 and length <= yloc_pixel + 5:
                        foreground.putpixel((width, length), (0, 255, 0, 100))
                    elif pixel[0] == pixel[1] == pixel[2] == 255:
                        foreground.putpixel((width, length), (0, 0, 0, 100))
                    else:
                        foreground.putpixel((width, length), (255, 0, 0, 100))

            background.paste(foreground, (0, 0), foreground)
            axarr[i][j+1].imshow(np.array(background), cmap='gray')
            axarr[i][j+1].axis("tight")
            axarr[i][j+1].axis("image") 
            axarr[i][j+1].axis('off')

        # Ground Truth
        if show_ground_truth:
            for j in range(3):
                if j == 1:
                    mask = (lbl[:,:,j] == 8)
                    lbl[mask] = 0
                if j == 2:
                    mask = (lbl[:,:,j] == 9)
                    lbl[mask] = 0
                axarr[i][j+4].imshow(lbl[:,:,j], cmap='gray')
                axarr[i][j+4].axis("tight")
                axarr[i][j+4].axis("image") 
                axarr[i][j+4].axis('off')

    row_axarr = None
    col_axarr = None
    if show_ground_truth:
        cols = ['Env', 'Prediction CR', 'Prediction DOF1', 'Prediction DOF2', 'Ground Truth CR', 'Ground Truth DOF1', 'Ground Truth DOF2']
    else:
        cols = ['Env', 'Prediction CR', 'Prediction DOF1', 'Prediction DOF2']
    col_axarr = axarr[0,:]

    for ax, col in zip(col_axarr, cols):
        ax.set_title(col)
    f.tight_layout()

    if show_ground_truth:
        fig_filepath = os.path.join('results', model_name, 'predictions-gt.png')
    else:
        fig_filepath = os.path.join('results', model_name, 'predictions.png')

    f.savefig(fig_filepath)
    plt.show()

def plot_activations(model_name):

    dataPath = os.path.join('results', model_name)
    samplefiles = glob.glob(dataPath+'/*[0-9].npy')
    f, axarr = plt.subplots(len(samplefiles), 12, gridspec_kw = {'wspace':0, 'hspace':0})
    for i, samplefile in enumerate(samplefiles):
        sample = os.path.split(samplefile)[-1]
        print("processing {sample}".format(sample=sample))
        envnum = sample[:-4]

        # load env image to compare with label pixel by pixel
        inp = np.load(os.path.join('input', 'raw', 'test', envnum + '.npy'))
        pred = np.load(os.path.join('results', model_name, envnum + '.npy'))
        logits = np.load(os.path.join('results', model_name, envnum + '_logits.npy'))
        
        pred = pred.reshape((224,224,3))
        logits = logits.reshape((224,224,20))

        def sigmoid(x):
            s = 1/(1+np.exp(-x))
            return s
        
        activations_cr = sigmoid(logits[:,:,0])
        prediction_cr = activations_cr.copy()
        mask = (activations_cr >= 0.5)
        prediction_cr[mask] = 1
        prediction_cr[~mask] = 0
        
        activations_dof1 = softmax(logits[:,:,1:10], axis=2)
        activations_dof2 = softmax(logits[:,:,10:], axis=2)


        axarr[i][0].imshow(inp[:,:,0], cmap='gray')
        axarr[i][0].axis("tight")
        axarr[i][0].axis("image") 
        axarr[i][0].axis('off')
        axarr[i][0].set_xticklabels([])
        axarr[i][0].set_yticklabels([])

        axarr[i][1].imshow(prediction_cr, cmap='gray')
        axarr[i][1].axis("tight")
        axarr[i][1].axis("image") 
        axarr[i][1].axis('off')
        axarr[i][1].set_xticklabels([])
        axarr[i][1].set_yticklabels([])

        axarr[i][2].imshow(activations_cr, cmap='gray')
        axarr[i][2].axis("tight")
        axarr[i][2].axis("image") 
        axarr[i][2].axis('off')
        axarr[i][2].set_xticklabels([])
        axarr[i][2].set_yticklabels([])
        for j in range(9):
            axarr[i][j+3].imshow(activations_dof1[:,:,j], cmap='gray')
            axarr[i][j+3].axis("tight")
            axarr[i][j+3].axis("image") 
            axarr[i][j+3].axis('off')
            axarr[i][j+3].set_xticklabels([])
            axarr[i][j+3].set_yticklabels([])

    cols = ['Env', 'CR','CR activation', '-pi', '-3pi/4', '-pi/2', '-pi/4', '0', 'pi/4', 'pi/2', '3pi/4']
    for ax, col in zip(axarr[0,:], cols):
        ax.set_title(col)
    
    f.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join('results', model_name, 'activations.png'))
    plt.show()

if __name__ == "__main__":
    
    model_name = 'unet-v26'
    print("Generating activation plots..")
    plot_activations(model_name)
    print("Generating prediction plots..")
    # plot_predictions(model_name, show_ground_truth=False)        
