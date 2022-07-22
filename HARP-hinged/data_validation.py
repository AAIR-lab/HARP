import numpy as np
import os



if __name__ == "__main__":
    dataPath = 'data'
    envlist = ['env17.2'] # sorted(os.listdir(dataPath))
    for env in envlist:
        envPath = os.path.join(dataPath, env)
        for category in ['inp', 'lbl']:
            path = os.path.join(envPath, category)
            samplefiles = sorted(os.listdir(path))
            for sample in samplefiles: 
                print("processing sample: {category}/{sample}".format(category=category, sample=sample))

                if sample.endswith('.npy') and category == 'inp': # augment input
                    inpfilePath = os.path.join(path, sample)
                    inp = np.load(inpfilePath)

                    assert np.sum(np.isnan(inp)) == 0
                    assert len(inp.shape) == 3 
                    assert inp.shape[2] == 7 # should have 7 channels (0-2: env image, 3-7: goal dof values)
                    for x in range(inp.shape[2]): # all channels should be normalized
                        assert inp[:,:,x].min() >= 0 and inp[:,:,x].max() <= 1.0
                
                elif sample.endswith('.npy') and category == 'lbl': # augment label
                    lblfilePath = os.path.join(path, sample)
                    lbl = np.load(lblfilePath)

                    assert np.sum(np.isnan(lbl)) == 0
                    assert len(lbl.shape) == 3
                    assert lbl.shape[2] == 3 # should have 3 channels (0: critical region gaze, 1-2: dof values)
                    for x in range(lbl.shape[2]): # all channels should be normalized
                        assert lbl[:,:,x].min() >= 0 and lbl[:,:,x].max() <= 1.0

print("Done")