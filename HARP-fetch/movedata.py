import pdb
import os
import shutil

'''
move inputs to `src/input/raw/train/`
move labels to `src/input/raw/train-labels/`
'''

dataPath = 'data'
networkTrain = os.path.join('network', 'input', 'raw', 'train')
networkTrainLabels = os.path.join('network', 'input', 'raw', 'train-labels')

envs = ['env17.2']

for env in os.listdir(dataPath): 
	envPath = os.path.join(dataPath, env)
	for category in ['inp', 'lbl']:
		path = os.path.join(envPath, category)
		print("processing: {env}/{category}".format(env=env, category=category))
		for sample in os.listdir(path): 
			samplepath = os.path.join(path, sample)
			if category == 'inp':
				shutil.move(samplepath, networkTrain)
			elif category == 'lbl':
				shutil.move(samplepath, networkTrainLabels)