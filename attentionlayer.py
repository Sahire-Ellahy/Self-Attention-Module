import math
import torch
import torch.nn as nn
import numpy as np

#computational graph issues?

# class MyAttentionLayer(nn.Module):
#     """ Custom attention layer made after studying "Attention is All you Need" paper and attention mechanism """
#     def __init__(self, Query_size, Key_size, Value_size):
#         super().__init__()
#         self.Query_size, self.Key_size, self.Value_size = Query_size, Key_size, Value_size
#         weights = torch.randn(Query_size, Key_size)
#         self.weights = nn.Parameter(weights)
    
#     def forward(self, Query, Keys, Values):
#         Query = torch.sum(Query, dim=0)
#         Query = torch.reshape(Query, (51, 1))
#         Keys = Keys.tolist() #do everything while avoiding using lists
#         Values = Values.tolist()
#         similarities = []
#         for key in Keys:
#             key = torch.Tensor(key)
#             weightedQuery = torch.reshape(torch.matmul(self.weights, Query), (1, 51))
#             similarity = torch.matmul(weightedQuery, key)
#             similarities.append(similarity)
#         similarities = torch.Tensor(similarities)
#         softmax = nn.Softmax(dim=0) #maybe should normalize input to length 1 before softmax?
#         coefficients = softmax(similarities)
#         attentionVect = torch.zeros(len(Values[0]))
#         for i in range(0,len(Values)):
#             attentionVect = attentionVect + torch.mul(torch.Tensor(Values[i]), coefficients[i])
#         return attentionVect




#is the RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn issue here??? something about the first operation of forward not having a grad function??

class MyAttentionLayer(nn.Module):
    """ Custom attention layer made after studying "Attention is All you Need" paper and attention mechanism """
    # def __init__(self, Query_size, Key_size, Value_size):
    #     super().__init__()
    #     self.Query_size, self.Key_size, self.Value_size = Query_size, Key_size, Value_size
    #     weights = torch.randn(Query_size, Key_size, requires_grad=True)
    #     self.weights = nn.Parameter(weights)
    #     # self.weights.requires_grad_()
    #     print(f"self.weights.requires_grad is {self.weights.requires_grad}")

    def __init__(self, QKV_size):
        super().__init__()
        self.QKV_size = QKV_size
        weights = torch.randn(QKV_size, QKV_size, requires_grad=True)
        self.weights = nn.Parameter(weights)
    
    # def forward(self, Query, Keys, Values): #Query, Keys, and Values are the same thing
    #     Query.requires_grad_()
    #     Query = torch.sum(Query, dim=0)
    #     weightedQuery = torch.matmul(self.weights, Query)
    #     similarities = torch.matmul(Keys, weightedQuery)
    #     weightedVects = Values * similarities[:, None]
    #     selfAttVect = torch.sum(weightedVects, dim=0)
    #     return selfAttVect

    def forward(self, QKV): #Query, Keys, and Values are the same thing and represented by QKV
        # QKV.requires_grad_()
        # KV = QKV.clone().detach() #need to use a copy of QKV; https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor is relevant
        KV = QKV.clone() #need to use a copy of QKV; https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor is relevant
        QKV = torch.sum(QKV, dim=0)
        # print(f"self.weights is on cuda: {self.weights.is_cuda} and QKV is on cuda: {QKV.is_cuda}")
        weightedQuery = torch.matmul(self.weights, QKV)
        # weightedQuery = torch.mm(self.weights, QKV)
        similarities = torch.matmul(KV, weightedQuery)
        # similarities = torch.mm(KV, weightedQuery)
        weightedVects = KV * similarities[:, None]
        selfAttVect = torch.sum(weightedVects, dim=0)
        return selfAttVect


#Works! Just need to replace labels with one hot vectors