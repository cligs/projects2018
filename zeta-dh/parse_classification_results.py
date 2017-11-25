from os.path import join
from pathlib import Path
import numpy as np
import re
import directory_paths

resultsfolder = directory_paths.resultsfolder
print("Using folder %s" % resultsfolder)

# data directory to store all values
collector = {}

# parse parameter string
def read_params(filename):
#results_50000-lemmata-all-rmul500-maxseg50_continent_America-Europe.txt_08
    m = re.search("results_(\d*)-lemmata-all-rmul500-maxseg(\d*)_continent_America-Europe.txt_(\d*)", filename)
    segments = m.group(1)
    maxsegs = m.group(2)
    run = m.group(3)
    return segments, maxsegs


for file in Path(resultsfolder).iterdir():
    file = str(file)

    # filtering dirs
    if Path(file).is_dir():
        continue

    # filtering files not following naming convention
    if not re.match(".*results_(\d*)-lemmata-all-rmul500-maxseg(\d*)_continent_America-Europe.txt_(\d*)", file):
        continue

    key = "%s-%s" % read_params(file)
    if key not in collector:
        collector[key] = {}
    with open(file, "r") as resfile:
        data = {}
        lastlabel = ""
        for line in resfile:
            #print(line)
            if not (line.startswith("[ ") or "Using" in line):
                continue
            if "Using" in line:
                field = line.split(" ")
                if len(field) == 12:
                    lastlabel = field[8]
                else:
                    lastlabel = "tfidf"
                if lastlabel.endswith("X"): # we don't use the ..X measures
                    continue
                #print(lastlabel)
            elif line.startswith("["):
                if lastlabel.endswith("X"):  # we don't use the ..X measures
                    continue

                if lastlabel not in collector[key]:
                    collector[key][lastlabel] = []
                #[ 0.5625  0.8125  0.4375]
                m = re.search("\[ (\d\.\d*)\s*(\d\.\d*)\s*(\d\.\d*)\s*\]", line)
                folds =  [float(m.group(1) + "0"), float(m.group(2) + "0"),float(m.group(3) + "0")]
                #print(folds)
                collector[key][lastlabel] += folds

if len(collector) == 0:
    raise Exception("No Files found")

measures = []
params = []

whichl = ["median", "mean", "min", "max"]

for which, lab in enumerate(whichl):
    # prepare matrix for data
    matrix = np.zeros((len(collector.keys()), 9))
    for i,k in enumerate(sorted(collector.keys())):
        params.append(k.replace("-", "_"))
        measures = sorted(collector[k].keys())
        for j,m in enumerate(measures):
            median = np.median(collector[k][m])
            mean = np.mean(collector[k][m])
            min = np.min(collector[k][m])
            max = np.max(collector[k][m])
            vals = [median, mean, min, max]
            matrix[i, j] = vals[which]

    #print(matrix)
    x,y = matrix.shape
    print(matrix[0,0])
    print(matrix[x-1,y-1])
    print(measures)
    matrix = matrix.transpose()


    #import matplotlib.pyplot as plt
    #plt.imshow(matrix, cmap='hot', interpolation='nearest')
    #plt.show()




    import plotly as py
    import plotly.graph_objs as go


    layout = go.Layout(
        title='Parameter heatmap for different zeta measures',
        xaxis=dict(
            title='Parameters (segment-size / samples)'
        ),
        yaxis=dict(
            title=lab + ' F1 for measure'
        )
    )

    trace = go.Heatmap(z=matrix, x=params, y=measures)
    data=[trace]
    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig, filename='measures-%s-f1' % lab)
