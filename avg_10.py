import sys
f = file(sys.argv[1], 'r').read().split('\n')

avg = 0
for line in f[-10:]:
	avg += float(line)

print(avg/10)