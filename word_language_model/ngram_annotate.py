import kenlm
import sys
import pdb

outfile = open(sys.argv[3], 'w')

print("Loading language model")
lm = kenlm.LanguageModel(sys.argv[1])
on = 0
for line in open(sys.argv[2], 'r').readlines():
    on += 1
    if on % 100 == 0:
        print("Currently on", on)
    line = line.strip()
    line = line.split('\t')
    orig= lm.score(line[0])
    # para= lm.score(line[4])
    # pdb.set_trace()
    perp = str(lm.perplexity(line[0]))
    # pdb.set_trace()
    # line += [str(orig - para)]
    line += [str(orig), str(perp)]
    outfile.write('\t'.join(line) + '\n')

