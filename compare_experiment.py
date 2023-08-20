import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import matplotlib.pylab as pylab

parser = ArgumentParser()
parser.add_argument('--compared_metric', type=str, choices=['rmse','r2','mae','evs','mape'], 
                                          help='Choose a metric to be compared', default='rmse')
parser.add_argument('--compared_model', type=str, choices=['bert','distil_roberta'], 
                                          help='Choose a model to be compared', default='bert')

args = parser.parse_args()

# Creating dataset
data_extra = np.load(f'{args.compared_metric}_extra_{args.compared_model}.npy')
data_inter = np.load(f'{args.compared_metric}_inter_{args.compared_model}.npy')
data_rf = np.load(f'{args.compared_metric}_rf_verify_{args.compared_model}.npy')

data = [data_extra, data_inter, data_rf]
# fig = plt.figure(figsize =(10, 7),dpi=250)
pylab.rcParams['font.size'] = 16
fig = plt.figure(figsize=(8,3),dpi=280)
ax = fig.add_subplot(111)
 
# Creating axes instance
bp = ax.boxplot(data, patch_artist = True,
                notch ='True', 
                vert = 0
                )
 
colors = ['#0000FF', '#00FF00', '#FFFF00']
 
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
 
# changing color and linewidth of
# whiskers
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")
 
# changing color and linewidth of
# caps
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
 
# changing color and linewidth of
# medians
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)
 
# changing style of fliers
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
     
# x-axis labels
ax.set_yticklabels(['Extrapolation','Interpolation','Adversarial\nTraning'])
 
# Adding title
metric = {'rmse':'RMSE','r2':'R2','mae':'MAE','evs':'EVS','mape':'MAPE'}
model = 'BERT' if args.compared_model=='bert' else 'RoBERTa'
plt.title(f"{metric[args.compared_metric]} between experiments for {model}")
 
# Removing top axes and right axes
# ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# save and show plot
plt.tight_layout()
plt.savefig(f'image/compare_experiments/compare_{args.compared_metric}_{args.compared_model}.png')
plt.show()