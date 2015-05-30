import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')


#read in tsne data
f = file('tsne.txt', 'r').read().split('\n')

# plt.subplots_adjust(bottom = 0.1)



datax = []
datay = []

for line in f[:-1]:
	label, x, y = line.split()
	if label in STOPWORDS: continue
	x = float(x)
	y = float(y)
	datax.append(x)
	datay.append(y)
	plt.annotate(label, xy = (x, y), xytext = (0, 0), textcoords = 'offset points')

plt.scatter(datax, datay, color='white')
plt.savefig('tsne.pdf')
# plt.show()