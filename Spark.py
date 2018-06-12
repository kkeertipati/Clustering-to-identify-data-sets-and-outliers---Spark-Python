from __future__ import print_function

import numpy as np
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from pyspark.sql.functions import countDistinct


sc = SparkContext(appName="Group_MIK")
sqlContext = SQLContext(sc)

# Set up Spark Session using builder
spark = SparkSession \
    .builder \
    .appName("any_name") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Read CSV - address is hardcoded here; Can also be set up to be passed as an argument


df = spark.read.format('csv').\
    options(inferSchema='true').\
    load("hdfs://xxx.csv", separator=",")

# Change names
df = df.selectExpr("_c0 as pres", "_c1 as temp")

# Remove 'NA' or '0' - incorrect entries
df = df.dropna(subset=('pres', 'temp'), how="all")
df = df.filter(df['pres'] > 0)


vecAssembler = VectorAssembler(inputCols=["pres", "temp"], outputCol="features")
df_kmeans = vecAssembler.transform(df).select('pres','temp','features')


k = 3
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()
cost = model.computeCost(df_kmeans)


print("Cluster Centers: ")
for center in centers:
    print(center)
    
print("Cost:", cost)

#Assign clusters to events

transformed = model.transform(df_kmeans).select( 'pres','temp','prediction')
transformed.show()

count =transformed.groupby('prediction').count()
count.show()

transformed.write.csv('cluster.csv')
