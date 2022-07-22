import PIL.Image as Image
import os
import matplotlib.pyplot as plt
import numpy as np
import wget
import shutil
import json
from bs4 import BeautifulSoup

'''
visualize training results from tensorboard api outputs
'''

host_url = 'http://10.218.111.197:6006' # 127.0.0.1 # 10.218.111.197
run = 'w5-z500-b32_2' # 'w5-z500'
step_index = 9
tags = ['input', 'prediction_cr', 'y_gaze']
num_images = 10
results = {}

# clean up tmp folder
if os.path.isdir(os.path.join('results', 'tmp')):
    shutil.rmtree(os.path.join('results', 'tmp'))
os.makedirs(os.path.join('results', 'tmp'))

# download images
for tag in tags:
    if tag not in results.keys():
        results[tag] = {}
    for i in range(num_images):
        url = host_url + r'/data/plugin/images/individualImage?index=' + str(step_index) + '&tag='+ tag + '%2Fimage%2F'+ str(i) + '&run=' + run
        print(url)
        filename = 'results/tmp/{index}_{tag_type}.png'.format(index=str(i), tag_type=tag)
        wget.download(url, filename)
        results[tag][i] = Image.open(filename)

# Get Goal conditions
url = host_url + r'/data/plugin/text/text?run=' + run + '&tag=FC_2%2Fgoal'
filename = 'results/tmp/{run}.json'.format(run=run)
wget.download(url, filename)
resp = json.load(open(filename, 'rb'))
html = resp[-1]['text']
soup = BeautifulSoup(html)
results['goals'] = {}
for i, tr_node in enumerate(soup.find_all('tr')[:num_images]):
    goal = [float(td_node.find_all('p')[0].text) for td_node in tr_node.find_all('td')]
    goal[0] = goal[0]*5 + (-2.5)
    goal[1] = goal[1]*5 + (-2.5)
    pw = 5./224.
    xloc_pixel = int((goal[0] + 2.5)/pw)
    yloc_pixel = int((goal[1] + 2.5)/pw)
    results['goals'][i] = (xloc_pixel, yloc_pixel)


f, axarr = plt.subplots(4, num_images, figsize=(20,20))

# Env Images
for j in range(num_images):
    axarr[0][j].imshow(results['input'][j])

# GT
for j in range(num_images):
    gaze = results['y_gaze'][j].convert("RGB")
    for width in xrange(gaze.size[0]):
        for length in xrange(gaze.size[1]):
            pixel = gaze.getpixel((width, length))
            if width >= results['goals'][j][0] - 5 and width <= results['goals'][j][0] + 5 and \
                length >= results['goals'][j][1] - 5 and length <= results['goals'][j][1] + 5:
                gaze.putpixel((width, length), (0, 255, 0))
            else:
                gaze.putpixel((width, length), (pixel[0], pixel[1], pixel[2]))
    axarr[1][j].imshow(np.array(gaze))

# Prediction
for j in range(num_images):
    axarr[2][j].imshow(results['prediction_cr'][j])

# Prediction + Env
for j in range(num_images):
    foreground = results['input'][j].convert("RGBA")
    # pred = np.array(results['prediction_cr'][j])
    background = results['prediction_cr'][j].convert("RGBA")
    for width in xrange(foreground.size[0]):
        for length in xrange(foreground.size[1]):
            pixel = foreground.getpixel((width, length))
            if width >= results['goals'][j][0] - 5 and width <= results['goals'][j][0] + 5 and \
                length >= results['goals'][j][1] - 5 and length <= results['goals'][j][1] + 5:
                foreground.putpixel((width, length), (0, 255, 0, 100))
            elif pixel[0] == pixel[1] == pixel[2] == 255:
                foreground.putpixel((width, length), (255, 255, 255, 100))
            else:
                foreground.putpixel((width, length), (255, 0, 0, 100))

    background.paste(foreground, (0, 0), foreground)
    axarr[3][j].imshow(np.array(background))

rows = ['Env', 'GT', 'Prediction', 'Prediction + Env']
for ax, row in zip(axarr[:, 0], rows):
    ax.set_ylabel(row)

f.tight_layout()

figpath = os.path.join('results','train', run + '.png')
f.savefig(figpath)
plt.show()
