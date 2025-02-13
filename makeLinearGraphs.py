import matplotlib.pyplot as plt # pip install -U matplotlib
import glob
import numpy as np # pip install numpy

def makeLinearGraph(lister, trainingLoops):
    plt.figure(figsize=(8,5), dpi=100) #matplotlib is great at making graphs, so start with defining the size
    plt.plot(lister, 'b--', label='accuracy')
    x2 = np.arange(1, len(lister) + 1)
    plt.title('Accuracy graph', fontdict={'fontname': 'Comic Sans MS', 'fontsize': 20})
    # X and Y labels
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
   
    plt.xticks(x2)
    plt.legend()

    counter = 1
    while glob.glob(f"model{counter}*.png"):
        counter+=1
    fileName = f"model{counter}_{trainingLoops}loopsPlot.png"
    plt.savefig(fileName)
    print(f"Plot saved as {fileName}")
    return fileName