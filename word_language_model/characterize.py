#!/usr/bin/env python3

import sys

for line in open(sys.argv[1], 'r'):
	print(' '.join(line.strip().replace(' ', '@')))
