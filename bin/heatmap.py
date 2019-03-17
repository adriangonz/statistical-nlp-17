from argparse import ArgumentParser
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
import tkinter
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument(
    "-r",
    "--attention",
    action="store",
    dest="attention",
    type=str,
    help="Path to the stored model's attentions")



def main(args):
	results = np.load(args.attention)
	attention = np.array(results['attention'])
	labels = np.array(results['labels'])
	support_set = np.array(results['support_set'])
	target_labels = np.array(results['target_labels'])
	targets = np.array(results['targets'])
	print(attention)
	print(labels)
	print(support_set)
	print(target_labels)
	print(targets)
	print(attention.shape)
	print(labels.shape)
	print(support_set.shape)
	print(target_labels.shape)
	print(targets.shape)


	data = attention.squeeze(0)
	fig, axis = plt.subplots() 
	heatmap = axis.pcolor(data, cmap=plt.cm.Blues) 
	
	# had to do it manually
	axis.set_yticks([0.5,1.5,2.5,3.5,4.5], minor=False)
	axis.set_xticks([0.5,1.5,2.5], minor=False)
	
	#same here
	column_labels = [1,2,3]	
	axis.invert_yaxis()
	plt.xlabel('Example (k)')
	plt.ylabel('Label (N)')
	axis.set_yticklabels(labels, minor=False)
	axis.set_xticklabels(column_labels, minor=False)
	plt.title('Attention map for examples in Support Set \n', fontsize=16)
	fig.set_size_inches(11.03, 7.5)
	plt.colorbar(heatmap)
	file_name = (f"{args.attention}_heatmap.png")
	plt.savefig(file_name, dpi=100)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
