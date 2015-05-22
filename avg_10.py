import sys
f = file(sys.argv[1], 'r').read().split('\n')

avg = 0
for line in f[-11:-1]:
	avg += float(line)

print(avg/10)