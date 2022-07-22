import os
import numpy as np
import time

if __name__ == "__main__":
    starttime = time.time()
    dataPath = 'input/raw'
    envlist = sorted(os.listdir(dataPath))
    for category in ['train']:
        path = os.path.join(dataPath, category)
        samplefiles = sorted(os.listdir(path))
        for sample in samplefiles: 
            print("processing sample: {category}/{sample}".format(category=category, sample=sample))

            if sample.endswith('.npy') and category == 'train': # augment input
                inpfilePath = os.path.join(path, sample)
                inp = np.load(inpfilePath)

                inp_tensor = inp.astype(np.float32)

                assert np.sum(np.isnan(inp)) == 0
                assert len(inp.shape) == 4
                assert inp.shape[-1] == 17 # should have 17 channels (0-2: env image, 3-10: goal dof, 11-16: fetch location)
                for x in range(inp.shape[-1]): # all channels should be normalized
                    assert inp[:,:,x].min() >= 0 and inp[:,:,x].max() <= 1.0 # [0,1]
                    # assert inp[:,:,x].min() >= -1.0 and inp[:,:,x].max() <= 1.0 # [-1,1]

                np.save(inpfilePath, inp_tensor)

    print("Completed in {runtime}mins".format(runtime=(time.time()-starttime)/60.0))