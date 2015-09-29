''' plot R, Q, completion rate for multiple files at the same time'''

import sys, argparse
import matplotlib.pyplot as plt
import math

plt.gcf().subplots_adjust(bottom=0.15)

f = []
for i in range(1,len(sys.argv)):
	f.append(map(float, file(sys.argv[i]).read().split('\n')[1:-1]))

max_epochs = 100
N = min(max_epochs, min(map(len, f)))

colors = ['red', 'orange', 'b']
markers = ['x', 6, '.']
# linestyles = ['-', '--', '-.', ':']

linestyles = ['-', '-','-']
# labels = ['Random', 'BOW-DQN']
labels = ['LSTM-DQN', 'BI-DQN', 'BOW-DQN']
# labels = ['LSTM-DQN', 'BI-DQN', 'BOW-DQN', 'Random']
# labels = ['BI-DQN', 'BOW-DQN', 'BI-LIN', 'BOW-LIN']
# labels = ['No Transfer', 'Transfer']
# labels = ['Uniform', 'Prioritized']
for i in range(len(f)):
	plt.plot(f[i][:N], color=colors[i], label=labels[i], linestyle=linestyles[i], markersize=6, linewidth=3) #normal scale
	# plt.plot([-math.log(abs(x)) for x in f[i][:N]], color=colors[i], label=labels[i], linestyle=linestyles[i], markersize=6, linewidth=3) #log scale

plt.xlabel('Epochs', fontsize=20)

# plt.ylabel('Reward (log scale)', fontsize=25)
plt.ylabel('Reward', fontsize=25)
# plt.ylabel('Max Q', fontsize=20)
# plt.ylabel('Quest Completion', fontsize=20)

plt.legend(loc=4, fontsize=15)
labelSize=17
plt.tick_params(axis='x', labelsize=labelSize)
plt.tick_params(axis='y', labelsize=labelSize)


x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,y1,1.2)) #set y axis limit



plt.savefig('plots/plot.pdf')
