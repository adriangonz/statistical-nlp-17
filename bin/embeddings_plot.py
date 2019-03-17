from argparse import ArgumentParser
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter
from src.similarity import euclidean_similarity



parser = ArgumentParser()
parser.add_argument(
    "-e",
    "--embedding",
    action="store",
    dest="embedding",
    type=str,
    help="Path to the stored model's attentions")



def attention(support_embeddings, target_embeddings):
	similarities = euclidean_similarity(support_embeddings, target_embeddings)

	# Compute attention as a softmax over similarities
	_, T, N, k = similarities.shape
	flat_similarities = similarities.view(-1, T, N * k)
	flat_attention = F.softmax(flat_similarities, dim=2)
	attention = flat_attention.view(-1, T, N, k)

	return attention

def main(args):
	results = np.load(args.embedding)
	sup_embeddings = np.array(results['support_embeddings'])
	target_embeddings = np.array(results['target_embeddings'])
	labels = np.array(results['labels'])
	support_set = np.array(results['support_set'])
	target_labels = np.array(results['target_labels'])
	targets = np.array(results['targets'])

	print(sup_embeddings)
	print(target_embeddings)
	print(labels)
	print(support_set)
	print(target_labels)
	print(targets)

	print(sup_embeddings.shape)
	print(target_embeddings.shape)
	print(labels.shape)
	print(support_set.shape)
	print(target_labels.shape)
	print(targets.shape)

	

	#set font size of labels on matplotlib plots
	plt.rc('font', size=16)

	#set style of plots
	sns.set_style('white')

	#define a custom palette
	customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139', '#2CB739']
	sns.set_palette(customPalette)
	sns.palplot(customPalette)

	#number of points per group
	n = 50

	#define group labels and their centers
	groups = {labels[0]: (1,1),
		      labels[1]: (1,4),
		      labels[2]: (4,4),
		      labels[3]: (4,1),
			  labels[4]: (3,3)}

	similarity = attention(sup_embeddings, target_embeddings)
	print (similarity.shape)
	
##############################TO CHANGE COMPLETELy #############################
	#create labeled x and y data
	data = pd.DataFrame(index=range(n*len(groups)), columns=['x','y','label'])
	for i, group in enumerate(groups.keys()):
		#randomly select n datapoints from a gaussian distrbution
		data.loc[i*n:((i+1)*n)-1,['x','y']] = np.random.normal(groups[group], 
		                                                       [0.5,0.5], 
		                                                       [n,2])
		#add group labels
		data.loc[i*n:((i+1)*n)-1,['label']] = group

	data.head()

	#create a new figure
	plt.figure(figsize=(10,10))
	#loop through labels and plot each cluster
	for i, label in enumerate(groups.keys()):

		#add data points 
		plt.scatter(x=data.loc[data['label']==label, 'x'], 
		            y=data.loc[data['label']==label,'y'], 
		            color=customPalette[i], 
		            alpha=0.20)
		
		#add label
		plt.annotate(label, 
		             data.loc[data['label']==label,['x','y']].mean(),
		             horizontalalignment='center',
		             verticalalignment='center',
		             size=20, weight='bold',
		             color=customPalette[i]) 

	name = args.embedding[:-14] 
	name = name[8:]
	file_name = (f"{name}_heatmap.png")	
	plt.savefig(file_name)#, dpi=500)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
