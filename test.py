import torch
import torch.nn.functional as F
import math
from torch import nn
# import cv2
import csv
def img_Contrast(img):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    
    # -----Applying CLAHE to L-channel------------------------------------------- 
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 
    cl = clahe.apply(l) 
    # -----Merge the CLAHE enhanced L-channel with the a and b channel----------- 
    limg = cv2.merge((cl, a, b)) 
    # -----Converting image from LAB Color model to RGB model-------------------- 
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) 
    return final



# def point_sample(input, point_coords, **kwargs):
#     """
#     From Detectron2, point_features.py#19
#     A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
#     Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
#     [0, 1] x [0, 1] square.
#     Args:
#         input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
#         point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
#         [0, 1] x [0, 1] normalized point coordinates.
#     Returns:
#         output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
#             features for points in `point_coords`. The features are obtained via bilinear
#             interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
#     """
#     add_dim = False
#     if point_coords.dim() == 3:

#         print("point_coords.dim() == 3")
#         add_dim = True
#         point_coords = point_coords.unsqueeze(2)
#     output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
#     if add_dim:
#         output = output.squeeze(3)
#     return output

#[0,1]

# f = open('/home/hpclab/kyg/lpd/out/0321/test.csv','w', newline='')
# wr = csv.writer(f)

# result = []
# for i in range(10):
#     result = [str(i)+'.jpg',i+1]
#     wr.writerow(result)

 
# f.close()

################
# occ = torch.tensor([[1,0,1,1,0,1,1,1,0,1]])
# input = torch.tensor([[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]])
occ = torch.tensor([[1,0,1,1,0,1,1,1,0,1],[1,0,1,1,0,1,1,1,0,1]])
input_p = torch.tensor([[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]])
input_oc = torch.tensor([[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]])

# print("input",input.shape)
print("input_p",input_p.shape)
print("input_p",input_p)
print("input_oc",input_oc.shape)
print("input_oc",input_oc)
print("occ",occ.shape)
print("occ",occ)

print("len(input_p)",len(input_p))
k = 0.5
# print("input_oc[i] >= k",input_oc[0] >= k)
#input 중에서 k보다 크거나 같은 것들의 인덱스 모음

# aa = (input >= k).nonzero(as_tuple=True)[1] 
for i in range(len(input_oc)) :
    print()
    print(str(i)+'step -- ')
    aa = (input_oc[i] >= k).nonzero(as_tuple=True)[0]
    print(aa)
    
# print("aa",aa)

    selected = torch.index_select(input_p[i],dim=0,index=aa)
    print("selected",selected.shape)
    print("selected",selected)

    occ_selected = torch.index_select(occ[i],dim=0,index=aa)
    print("occ_selected",occ_selected.shape)
    print("occ_selected",occ_selected)
    print()


################

# t = torch.Tensor([1, 2, 3])
# print ((t == 2).nonzero(as_tuple=True)[0])

# selected = torch.index_select(input,dim=1,index=torch.LongTensor([0,1]))
# print("selected",selected.shape)
# print("selected",selected)

# x = torch.rand(2, 3, 4)
# y = torch.rand(2, 5)
# print("x",x.shape)

# print("y",y.shape)
# y = y.unsqueeze(1)

# print("y",y.shape)
# y=y.repeat(1,3,1)
# print("y",y.shape)
# z = torch.cat([x,y],dim=2)
# print("z",z.shape)
# print("x",x)
# x = -1*x +1
# print("x",x.shape)
# print("x",x)
# input=torch.tensor([[[[-0.9,  0.9],[-0.9, 0.9]]]])
# print("input",input.shape)
# print("input",input)
# x=torch.tensor([[[0.6, 0.5]]])
# x = torch.rand(1, 2, 2 , 2)
# x = torch.tensor([[[[0.6, 0.6],\
#           [0.5, 0.5]]]])
# print("x",x.shape)
# print("x",x)
# y = point_sample(input,x)
# print("y",y.shape)
# print("y",y)
