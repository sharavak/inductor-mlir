import torch
import torch.nn as nn


class CustomBatchNorm2d(nn.Module):
  def __init__(self, num_features, eps=1e-5, affine=True):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None


  def forward(self, x):
    B,C,W,H=x.shape
    batch_sum=x.sum(dim=0,keepdim=True)
    print(batch_sum.shape)
    height_sum=batch_sum.sum(dim=2,keepdim=True)
    print(height_sum.shape)
    width_sum=height_sum.sum(dim=3,keepdim=True)
    print(width_sum.shape)

    m=torch.tensor([B*W*H]) # creating the tensor since recriprocal op supports only tensor

    noOfEle=torch.reciprocal(m)

    width_avg=torch.mul(width_sum,noOfEle)

    var_sub=torch.sub(x,width_avg) #(x-mean)

    variance=torch.mul(var_sub,var_sub) # (x-mean)*(x-mean)
    print(variance.shape,'variance')

    batch_var=variance.sum(dim=(0,2,3),keepdim=True) # it is same as torch.sum
    print(noOfEle.shape,'noofelements shape')

    batch_var=torch.mul(batch_var,noOfEle)

    print(batch_var.shape,"batch shape");

    add=torch.add(batch_var,self.eps)

    print(add.shape,"shape")
    sq=torch.sqrt(add)

    sq=torch.reciprocal(sq)

    x_norm=torch.mul(var_sub,sq)

    return x_norm


for i in range(5):
    b,ch,w,h=torch.randint(100,size=(1,)),torch.randint(100,size=(1,)),torch.randint(100,size=(1,)),torch.randint(100,size=(1,))
    ip=torch.randn(b,ch,w,h)
    cus=CustomBatchNorm2d(ch,affine=False)
    res=cus(ip)
    batch_norm = nn.BatchNorm2d(ch, affine=False)
    print(torch.allclose(batch_norm(ip),res))