import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def getError(errorFactor, delta):
    error = errorFactor * np.random.randint(low=-1,high=2) * np.random.random_sample() * delta
    return error

def getSimDist (vec1, vec2):
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    if (mag1 > 0 and mag2 > 0):
        dist = np.linalg.norm(vec1 - vec2)
        similarity = np.dot(vec1,vec2) / (mag1 * mag2)
        return similarity, dist
    else:
        return -99999.0, -99999.0

def findMetrics (vectors):
    metrics = {}
    centroid = np.average(vectors, axis=1)
    nVectors = np.size(vectors,1)
    distances = []
    similarities = []
    for i in range(0, nVectors):
        sim, dist = getSimDist(centroid, vectors[:,i])
        if (dist > 0):
            similarities.append(sim)
            distances.append(dist)
    metrics['centroid'] = centroid
    metrics['distances'] = np.array(distances)
    metrics['similarities'] = np.array(similarities)
    metrics['similarity-stats'] = {'min' : np.amin(similarities), 'mean' : np.mean(similarities), 'median' : np.median(similarities), 'max' : np.amax(similarities), 'std' : np.std(similarities)}
    metrics['distance-stats'] = {'min' : np.amin(distances), 'mean' : np.mean(distances), 'median' : np.median(distances), 'max' : np.amax(distances), 'std' : np.std(distances)}

    return metrics

np.random.seed(42)
np.set_printoptions(precision=3)

theta1 = 15.0 * np.deg2rad(180.0) / 180.0
theta2 = 25.0 * np.deg2rad(180.0) / 180.0

args = sys.argv
strecthFactor = float(args[1])       # 1, or 2
angleFactor = float(args[2])         # 1, or 6

pb = strecthFactor * np.array([ [np.cos(angleFactor*theta1), np.cos(angleFactor*theta2)], [np.sin(angleFactor*theta1), np.sin(angleFactor*theta2)] ])
pbInv = np.linalg.inv(pb)

nPoints = 100
xmin = 0.1
xmax = 0.9
errorFactor = 5.0
x = np.linspace(xmin, xmax, nPoints)
dx = (xmax -xmin) / nPoints
vectors = {}
for item in ['c1_original', 'c2_original', 'c1_trasformed', 'c2_trasformed']:
    vectors[item] = np.zeros((2,nPoints))

for i in range (0, nPoints):
    vectors['c1_original'][0,i] = x[i] + getError(errorFactor, dx)
    vectors['c1_original'][1,i] = x[i] * np.tan(theta1) + getError(errorFactor, dx)
    vectors['c2_original'][0,i] = x[i] + getError(errorFactor, dx)
    vectors['c2_original'][1,i] = x[i] * np.tan(theta2) + getError(errorFactor, dx)

vectors['c1_trasformed'] = np.matmul(pbInv, vectors['c1_original'])
vectors['c2_trasformed'] = np.matmul(pbInv, vectors['c2_original'])

metrics = {}
for item in ['c1_original', 'c2_original', 'c1_trasformed', 'c2_trasformed']:
    metrics[item] = findMetrics (vectors[item])

medianMetrics = {}
for item in ['original', 'trasformed']:
    medianMetrics[item] = {}
    for metric in ['distance', 'similarity']:
        medianMetrics[item][metric] = (metrics['c1_'+item][metric+'-stats']['median'] + metrics['c2_'+item][metric+'-stats']['median'])/2.0

c1_c2_similarity, c1_c2_distance = getSimDist (metrics['c1_original']['centroid'] , metrics['c2_original']['centroid'])
c1_c2_similarity_transformed, c1_c2_distance_transformed = getSimDist (metrics['c1_trasformed']['centroid'] , metrics['c2_trasformed']['centroid'])

print ('c1 centroid, c2 centroid',metrics['c1_original']['centroid'], metrics['c2_original']['centroid'])
print ('c1_trasformed centroid, c2_transformed centroid',metrics['c1_trasformed']['centroid'], metrics['c2_trasformed']['centroid'])

print ('original medan similarity, transformed median similarity',medianMetrics['original']['similarity'], medianMetrics['trasformed']['similarity'])
print ('c1_c2_similarity, c1_c2_similarity_transformed',c1_c2_similarity, c1_c2_similarity_transformed)
print ('c1_c2_similarity_scaled, c1_c2_similarity_transformed_scaled',c1_c2_similarity/medianMetrics['original']['similarity'],c1_c2_similarity_transformed/medianMetrics['trasformed']['similarity'])

print ('original medan distance, transformed median distance',medianMetrics['original']['distance'], medianMetrics['trasformed']['distance'])
print ('c1_c2_distance, c1_c2_distance_transformed',c1_c2_distance,c1_c2_distance_transformed)
print ('c1_c2_distance_scaled, c1_c2_distance_transformed_scaled',c1_c2_distance/medianMetrics['original']['distance'],c1_c2_distance_transformed/medianMetrics['trasformed']['distance'])

xmin = min(np.amin(vectors['c2_original'][0,:]), np.amin(vectors['c1_original'][0,:]), np.amin(vectors['c1_trasformed'][0,:]), np.amin(vectors['c2_trasformed'][0,:])) * 1.01
ymin = min(np.amin(vectors['c2_original'][1,:]), np.amin(vectors['c1_original'][1,:]), np.amin(vectors['c1_trasformed'][1,:]), np.amin(vectors['c2_trasformed'][1,:])) * 1.01
xmax = max(np.amax(vectors['c2_original'][0,:]), np.amax(vectors['c1_original'][0,:]), np.amax(vectors['c1_trasformed'][0,:]), np.amax(vectors['c2_trasformed'][0,:])) * 1.01
ymax = max(np.amax(vectors['c2_original'][1,:]), np.amax(vectors['c1_original'][1,:]), np.amax(vectors['c1_trasformed'][1,:]), np.amax(vectors['c2_trasformed'][1,:])) * 1.01

plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.axes().set_aspect('equal')

plt.plot(vectors['c1_original'][0,:], vectors['c1_original'][1,:], color='g', markersize=1, marker='*', linestyle='')
plt.plot(vectors['c2_original'][0,:], vectors['c2_original'][1,:], color='r', markersize=1, marker='*', linestyle='')
plt.plot(metrics['c1_original']['centroid'][0], metrics['c1_original']['centroid'][1], color='k', markersize=3, marker='s', linestyle='')
plt.plot(metrics['c2_original']['centroid'][0], metrics['c2_original']['centroid'][1], color='k', markersize=3, marker='s', linestyle='')
plt.savefig('original.png', format='png', dpi=720)
plt.close()

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.axes().set_aspect('equal')
plt.plot(vectors['c1_trasformed'][0,:], vectors['c1_trasformed'][1,:], color='g', markersize=1, marker='>', linestyle='')
plt.plot(vectors['c2_trasformed'][0,:], vectors['c2_trasformed'][1,:], color='r', markersize=1, marker='<', linestyle='')
plt.plot(metrics['c1_trasformed']['centroid'][0], metrics['c1_trasformed']['centroid'][1], color='k', markersize=3, marker='s', linestyle='')
plt.plot(metrics['c2_trasformed']['centroid'][0], metrics['c2_trasformed']['centroid'][1], color='k', markersize=3, marker='s', linestyle='')
plt.savefig('transformed.png', format='png', dpi=720)
plt.close()

dboxes = [ metrics['c1_original']['distances'], metrics['c1_trasformed']['distances'], metrics['c2_original']['distances'], metrics['c2_trasformed']['distances'] ]
plt.ylim(0.0, 0.6)
plt.boxplot(dboxes, showmeans=True,sym='',whis=[5,95])
plt.savefig('distances.png', format='png', dpi=720)
plt.close()


