!vw --version

import pandas as pd
df=pd.read_csv('trainset.csv',dtype=pd.StringDtype())

df['label'] = df['label'].replace(['1','0'],['+1','-1'])

df.head()

# df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.head()

def rem(text):
    return ''.join(text).replace(':','COLON').replace('|','PIPE')
formatted=df.clean_tweet
with open("train.txt",'w') as t:
    for i in formatted:
        t.write(rem(i)+'\n')


l=[]
f=open("train.txt",'r')
for i,j in zip(df.label,f.readlines()):
    l.append(i+' | '+j)
print(len(l))

import random
random.seed(1234)
random.shuffle(l)   # this does in-place shuffling
# print out the labels of the first 50 examples to be sure they're sane:
print (''.join([s[0] for s in l[:50]]))


def writeToVWFile(filename, examples):
    with open(filename, 'w') as h:
        for ex in examples:
            h.write(ex.strip()+'\n')
            
writeToVWFile('sentiment.tr', l[:2500])
writeToVWFile('sentiment.te', l[2500:])

!wc -l sentiment.tr sentiment.te

!vw --binary sentiment.tr --passes 20 -c -k -f sentiment.model --loss_function quantile --ngram 2

!ls -l sentiment.model

!vw --binary -t -i sentiment.model -p sentiment.te.pred sentiment.te


!vw --binary sentiment.tr --passes 20 -c -k -f sentiment.model --ngram 2 --loss_function quantile --quiet
!vw -i sentiment.model -t --invert_hash sentiment.model.readable sentiment.tr --quiet

!head -n40 sentiment.model.readable

!cat sentiment.model.readable  | tail -n+12 | sort -t: -k3nr | head

!cat sentiment.model.readable  | tail -n+12 | sort -t: -k3nr | tail

!echo "Top positive features"
!cat sentiment.model.readable  | tail -n+13 | sort -t: -k3nr | head
!echo ""
!echo "Top negative features"
!cat sentiment.model.readable  | tail -n+13 | sort -t: -k3nr | tail


