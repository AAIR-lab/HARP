import os
import numpy as np
import time
import sys

mode = sys.argv[1]

if __name__ == "__main__":
    starttime = time.time()
    dataPath = 'input/raw'
    envlist = sorted(os.listdir(dataPath))
    if mode == "train":
        for category in ['train', 'train-labels']:  # iterate over all inputs and labels
            path = os.path.join(dataPath, category)
            samplefiles = sorted(os.listdir(path))
            for sample in samplefiles: 
                print("processing sample: {category}/{sample}".format(category=category, sample=sample))

                if sample.endswith('.npy') and category == 'train':
                    inpfilePath = os.path.join(path, sample)
                    inp = np.load(inpfilePath)
                    
                    # Perform any processing here
                    inp_tensor = inp.astype(np.float32)
                    inp_tensor = inp_tensor[:,:,:3].reshape((224,224,3))

                    # Validation
                    assert np.sum(np.isnan(inp_tensor)) == 0
                    assert len(inp_tensor.shape) == 3
                    assert inp_tensor.shape[-1] == 7 # should have 17 channels (0-2: env image, 3-8: goal location, 9-14: fetch location)
                    # for x in range(inp_tensor.shape[-1]): # all channels should be normalized
                    #     assert inp_tensor[:,:,x].min() >= 0 and inp_tensor[:,:,x].max() <= 1.0 # [0,1]
                    
                    np.save(inpfilePath, inp_tensor)
                
                elif sample.endswith('.npy') and category == 'train-labels':
                    lblfilePath = os.path.join(path, sample)
                    lbl = np.load(lblfilePath)

                    # Perform any processing here
                    lbl_tensor = lbl.astype(np.float32)
                    # mask = lbl_tensor[:,:,:,0] < 0.2
                    # lbl_tensor[mask] = 0
                    # lbl_tensor[~mask] = 1
                    lbl_tensor = lbl_tensor[:,:,0].reshape((224,224,1))

                    # Validation
                    assert np.sum(np.isnan(lbl_tensor)) == 0
                    assert len(lbl_tensor.shape) == 3
                    assert lbl_tensor.shape[-1] == 21 # should have 1 channels
                    # for x in range(lbl_tensor.shape[-1]): # all channels should be normalized
                    #     assert lbl_tensor[:,x].min() >= 0 and lbl_tensor[:,x].max() <= 1.0 # [0,1]
                    
                    np.save(lblfilePath, lbl_tensor)

    if mode == "test":
        for category in ['test', 'test-labels']:  # iterate over all inputs and labels
            path = os.path.join(dataPath, category)
            samplefiles = sorted(os.listdir(path))
            for sample in samplefiles: 
                print("processing sample: {category}/{sample}".format(category=category, sample=sample))

                if sample.endswith('.npy') and category == 'test':
                    inpfilePath = os.path.join(path, sample)
                    inp = np.load(inpfilePath)
                    
                    # Perform any processing here
                    inp_tensor = inp.astype(np.float32)
                    # inp_tensor = inp_tensor[:,:,:3].reshape((224,224,3))

                    # Validation
                    assert np.sum(np.isnan(inp_tensor)) == 0
                    assert len(inp_tensor.shape) == 3
                    print inp_tensor.shape
                    assert inp_tensor.shape[-1] == 7 # should have 17 channels (0-2: env image, 3-8: goal location, 9-14: fetch location)
                    # for x in range(inp_tensor.shape[-1]): # all channels should be normalized
                    #     assert inp_tensor[:,:,x].min() >= 0 and inp_tensor[:,:,x].max() <= 1.0 # [0,1]
                    
                    np.save(inpfilePath, inp_tensor)
                
                elif sample.endswith('.npy') and category == 'test-labels':
                    lblfilePath = os.path.join(path, sample)
                    lbl = np.load(lblfilePath)

                    # Perform any processing here
                    lbl_tensor = lbl.astype(np.float32)
                    # mask = lbl_tensor[:,:,:,0] < 0.2
                    # lbl_tensor[mask] = 0
                    # lbl_tensor[~mask] = 1

                    # Validation
                    assert np.sum(np.isnan(lbl_tensor)) == 0
                    assert len(lbl_tensor.shape) == 3
                    assert lbl_tensor.shape[-1] == 21 # should have 1 channels
                    # for x in range(lbl_tensor.shape[-1]): # all channels should be normalized
                    #     assert lbl_tensor[:,x].min() >= 0 and lbl_tensor[:,x].max() <= 1.0 # [0,1]
                    
                    np.save(lblfilePath, lbl_tensor)

    print("Completed in {runtime}mins".format(runtime=(time.time()-starttime)/60.0))