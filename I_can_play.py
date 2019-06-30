from __future__ import division, print_function

import os
import sys
import argparse

import tempfile
import urllib

import numpy as np
import pandas as pd


class A(object):
    def __init__(self):
        pass

    def load_data(self, train_dfn="/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/eval_ins_continues.txtlineNUM_videoID",
                  eval_dfn="/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/train_ins_continues.txtlineNUM_videoID"):
        self.train_data = pd.read_csv(train_dfn, delim_whitespace=True, header=None)
        self.eval_data = pd.read_csv(eval_dfn, delim_whitespace=True, header=None)

        # self.train_data = self.train_data.sample(frac=1).reset_index(drop=True) #shuffle [以100%de比例随机选择原来的数据，drop=True自动新建一列记录原来的index]
        # self.eval_data = self.eval_data.sample(frac=1).reset_index(drop=True)

        self.train_data[self.label_column] = (self.train_data[self.label_column]).astype(int)
        self.eval_data[self.label_column] = (self.eval_data[self.label_column]).astype(int)

        self.train_data = self.train_data[self.train_data[self.label_column].isin([0,1])] #防止label列，即第0列，有除了0，1之外的值出现。
        self.eval_data = self.eval_data[self.eval_data[self.label_column].isin([0,1])]

    def prepare_input_data(self, input_data):
        X = input_data.iloc[:, 1:input_data.shape[1] - 1].values.astype(np.float32)
        Y = input_data[self.label_column].values.astype(np.float32)
        # print("X.shape = ",X.shape)
        # print("Y.shape = ",Y.shape)
        # X.shape = (50000, 108)
        # Y.shape = (50000,)
        Y = Y.reshape([-1, 1])
        if self.verbose:
            print("  Y shape=%s, X shape=%s" % (Y.shape, X.shape))
        X_dict = {"wide_X": X}
        Y_dict = {"Y": Y}
        import pdb
        pdb.set_trace()

        return X_dict, Y_dict

    def prepare(self):
        self.X_dict, self.Y_dict = self.prepare_input_data(self.train_data)
        self.evalX_dict, self.evalY_dict = self.prepare_input_data(self.eval_data)



if __name__ == "__main__":
    a = A()
    a.load_data()
    a.prepare()
