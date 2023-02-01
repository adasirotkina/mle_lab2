from pyspark.context import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
import numpy as np
import pandas as pd
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.feature import Normalizer
from pyspark.mllib.linalg.distributed import IndexedRowMatrix

from pyspark.sql.functions import *

import traceback

import os
from logger import Logger

SHOW_LOG = True

conf = SparkConf().set("spark.cores.max", "16") \
    .set("spark.driver.memory", "16g") \
    .set("spark.executor.memory", "16g") \
    .set("spark.executor.memory_overhead", "16g") \
    .set("spark.driver.maxResultsSize", "0")

sc = SparkContext('local')
spark = SparkSession(sc).builder.master("local[10]").config("spark.driver.memory", "10g").getOrCreate()

class TFIDF():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.project_path = os.path.join(os.getcwd(), "data")
        self.data_path = os.path.join(self.project_path, "trainx16x32_0.npz") #download data
        # self.log.info("Data is ready")


    def get_data(self):
        data_train = np.load(self.data_path)

        rdd = sc.parallelize(data_train['arr_0'][:256944])  # [:256944] #data to rdd
        rdd = rdd.groupByKey().mapValues(list) #words to sentences
        documents = rdd.map(lambda l: l[1])
        labels = rdd.map(lambda l: l[0]) #extract keys

        return documents, labels

    def matrix(self, documents, labels):
        try:
            # add TFIDF
            hashingTF = HashingTF()
            tf = hashingTF.transform(documents)

            tf.cache()
            idf = IDF().fit(tf)
            tfidf = idf.transform(tf)

            features = tfidf

            normalizer = Normalizer()
            data = labels.zip(normalizer.transform(features))

            #create matrix
            mat = IndexedRowMatrix(data).toBlockMatrix()
            dot = mat.multiply(mat.transpose())
            matrix = dot.toLocalMatrix().toArray()

            np.savetxt('matrix.txt', matrix)

            return matrix

        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)


if __name__ == "__main__":
    create = TFIDF()
    documents, labels = create.get_data()
    create.matrix(documents, labels)
    
    
sc.stop()