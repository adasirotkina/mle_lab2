import numpy as np
import argparse

from logger import Logger

SHOW_LOG = True

import random


class Predictor():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        #parse argument
        self.parser = argparse.ArgumentParser(description="Predictor")

        self.parser.add_argument("-m",
                                 "--mode",
                                 type=str,
                                 help="Select mode",
                                 required=True,
                                 default="1",
                                 const="1",
                                 nargs="?",
                                 choices=['random_n', 'by_id'])

        self.parser.add_argument("-id",
                                 "--input_id",
                                 type=int,
                                 help="Select film",
                                 required=False,
                                 default="0",
                                 const="0",
                                 nargs="?")

        self.parser.add_argument("-rc",
                                 "--random_count",
                                 type=int,
                                 help="Select random count",
                                 required=False,
                                 default="1",
                                 const="1",
                                 nargs="?")

        self.parser.add_argument("-c",
                                 "--count",
                                 type=int,
                                 help="Select reccomendation count",
                                 required=False,
                                 default="1",
                                 const="1",
                                 nargs="?")

        #load matrix
        self.matrix = np.loadtxt('matrix.txt', dtype=float)
        self.log.info("Predictor is ready")

    def predict(self):

        #add args
        args = self.parser.parse_args()

        if args.mode == 'by_id':
            dict_ = {i: self.matrix[args.input_id][i] for i in range(len(self.matrix[args.input_id]))}
            dict_sorted = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}

            return list(dict_sorted.keys())[1: args.count + 1]
        else:
            answer = {}
            ids = random.sample(list(range(len(self.matrix))), args.random_count)
            for id in ids:
                dict_ = {i: self.matrix[id][i] for i in range(len(self.matrix[id]))}
                dict_sorted = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}
                answer[id] = list(dict_sorted.keys())[1: args.count + 1]

            #write result
            with open(f'{args.mode}_RandomCount{args.random_count}_RecCount{args.count}.txt', 'w') as f:
                for k in answer.keys():
                    f.write(f'{k}: {answer[k]} \n')

            return answer



if __name__ == "__main__":
    predictor = Predictor()
    print(predictor.predict())


# def recomendation(matrix, input_id, count):
#
#
# # input_id = 42
# # count = 3
# # print(recomendation(matrix, input_id, count))
#
# rec_dict = {}
# for i in range(len(matrix)):
#     rec_dict[i] = recomendation(matrix, i, 1)
#
# print(rec_dict)