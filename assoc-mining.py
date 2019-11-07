from pyspark import SparkConf
from pyspark.context import SparkContext
sc = SparkContext(master='local[*]')
import sys
from itertools import combinations
sc.setLogLevel("ERROR")

#Importing the 'orders' dataset with two columns, viz (orderID, productID)
orders = sc.textFile('file:///home/cloudera/BDA/data/orders-small.csv')

#Filtering out blank lines
orders = orders.filter(lambda line: len(line)!=0)

#Forming key value pairs from the dataset (productID, orderID) using the vertical data format in Apriori
orders = orders.map(lambda line: (int(line.split(',')[1]), int(line.split(',')[0]) ) )

#Grouping values i.e orderID's by key i.e productID
#E.g: (23,67), (23,45), (23,78) => (23, (67,45,78)) where 23 is productID and 67,45 and 78 are transactionIDs
orders = orders.groupByKey().map(lambda line: (line[0], tuple([x for x in line[1]])  ))

#Importing the 'products' dataset with two columns, viz (productID, productName)
products = sc.textFile('file:///home/cloudera/BDA/data/products-small.csv')

#Filtering out blank lines
products = products.filter(lambda line: len(line)!=0)

#Forming key value pairs from the dataset (productID, productName)
products = products.map(lambda line: (int(line.split(',')[0]), line.split(',')[1].encode('utf-8').strip() ) )

#rule function prints the confidence value for association c => d
def rule(c,d):

	denom = orders.filter(lambda line: line[0] in c)
	denom = denom.map(lambda line: ('key', line[1]) )
	denom = denom.reduceByKey(lambda x,y: tuple( set(x).intersection(set(y)) )  )
	denom = denom.map(lambda line: ('key', line[1]))
	dcount = list(denom.collect())
	j = [products.lookup(x)[0] for x in c]
	k = [products.lookup(x)[0] for x in d]
	s = str(tuple(j)) + ' => ' + str(tuple(k))
	if len(dcount)==0:
		r = 'Confidence: 0%'
		return 0
	dcount = len(dcount[0][1])
	num = orders.filter(lambda line: line[0] in d)
	num = num.map(lambda line: ('key', line[1]) )
	num = num.union(denom)
	num = num.reduceByKey(lambda x,y: tuple( set(x).intersection(set(y)) )  )
	ncount = list(num.collect())
	ncount = len(ncount[0][1])
	if dcount!=0:
		r = 'Confidence: '+ str( round( (ncount*100)/float(dcount),2 ) ) + '%'	
		print(s+"\n"+r+"\n")

#Given a list of productIDs, getcomb functions returns the association combinations.
#E.g: For (1,2,3), getcomb returns the following associations
#(1) => (2,3)
#(2) => (1,3)
#(3) => (1,2)
#(1,2) => (3)
#(1,3) => (2)
#(2,3) => (1)

def getcomb(record):
    left = []
    right = []
    for i in range(1,len(record)):
        comb = set(combinations(record,i))
        for c in comb:
                l = set(c)
                r = record.difference(l)
                left.append(list(l))
                right.append(list(r))
    return left, right

#getrules function calls rule function for each of the association combination returned by getcomb function
def getrules(rec):
	left, right = getcomb(rec)
	for i in range(len(left)):
		m = left[i]
		n = right[i]
		rule(m,n)

#Get the list of productIDs from command line in the list 'pids'
pids = [int(c) for c in sys.argv[1].split(',')]

rc = set(pids)

print("\n\n\nFetching Association rules...\n\n")
getrules(rc)





