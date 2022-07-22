from __future__ import division
from sys import argv
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sys import *
from numpy import *
from scipy.spatial import distance 
import os, pickle, pdb 

envnum = argv[1]
bounds = [[-2.5, -2.5], [2.5, 2.5]] # all random envs are built using the same base world, thus they have the same world bounds
if envnum == '9.1':
	bounds = [[-10, -10], [10, 10]]
pixelwidth = (bounds[1][0] - bounds[0][0]) / 224

def pixelBounds(pixel, pixelwidth=pixelwidth, bounds=bounds):
	'''
	obtain pixel bounds in terms of world coordinates
	'''
	pixminx = bounds[0][0] + (pixel[0] * pixelwidth)
	pixminy = bounds[1][1] - ((pixel[1] + 1) * pixelwidth)
	pixmaxx = bounds[0][0] + ((pixel[0] + 1) * pixelwidth)
	pixmaxy = bounds[1][1] - (pixel[1] * pixelwidth)
	b = [(pixminx, pixminy), (pixmaxx, pixmaxy)]

	return b

def load_samples():
	with open(os.path.join('results','samples_env' + envnum +  '.pkl'), 'rb') as samples:
		gs = pickle.load(samples)

	return gs

def clusterSamples(samplelist):
	clusters = {} # key is cluster id and maps to list of indices of samples in that cluster
	pixelcentroids = []
	sampletrace = []

	for pixelbounds in samplelist:
		pixminx = pixelbounds[0]
		pixminy = pixelbounds[1]
		pixmaxx = pixelbounds[2]
		pixmaxy = pixelbounds[3]

		aveX = (pixmaxx + pixminx) / 2.0
		aveY = (pixmaxy + pixminy) / 2.0
		pixelcentroids.append([aveX,aveY])
	
	X = array(pixelcentroids)
	nbrs = NearestNeighbors(n_neighbors=25, algorithm='ball_tree').fit(X)
	distances, indices = nbrs.kneighbors(X)
	addedlist = list(indices[0])
	clusters[0] = list(indices[0])
	for i in xrange(indices.shape[0]):
		nearest = list(indices[i])
		added = False
		for clusternum in clusters.keys():
			if bool(set(nearest) & set(clusters[clusternum])):
				toadd = list(set(nearest)-set(addedlist))
				if toadd != []:
					clusters[clusternum] = clusters[clusternum] + toadd
					addedlist = addedlist + toadd
				added = True
				break
		if not added:
			clusters[len(clusters)] = nearest
			addedlist = addedlist + nearest

	return clusters.values(), pixelcentroids

def getPixelLocations():
	pixellocations = []
	modeloutput = Image.open('results/env'+envnum+'_output.png').convert("RGB")

	for length in xrange(modeloutput.size[1]):
		for width in xrange(modeloutput.size[0]):
			pixel = modeloutput.getpixel((width, length))

			if pixel == (255,255,255):
				pixellocations.append((width,length))

	return pixellocations

def main():
	clusterranks = []
	groundtruth = Image.open('../../datatest/env'+envnum+'/lbl/'+envnum+'0.png').convert("RGB")
	samples = load_samples()
	pixellocations = getPixelLocations()
	clusters, pixelcentroids = clusterSamples(samples)

	for cluster in clusters:
		rank = 0
		for landmarkID in cluster:
			rank += (groundtruth.getpixel(pixellocations[landmarkID])[0] / 255)
		clusterrank = rank / len(cluster)
		clusterranks.append(clusterrank)
	sumrank = sum(clusterranks)
	averank = sumrank / len(clusterranks)
	
	print 'cluster ranks: ', clusterranks
	print 'sum ranks: ', sumrank
	print 'average rank: ', averank

if __name__ == "__main__":
    main()