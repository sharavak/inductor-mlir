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
    # General Formula -> (x-mean)/sqrt(variance+eps)
    # Variance Formula -> (x-mean)*(x-mean)/N
    
    B,C,W,H=x.shape

    batch_sum=x.sum(dim=0,keepdim=True)
    height_sum=batch_sum.sum(dim=2,keepdim=True)
    width_sum=height_sum.sum(dim=3,keepdim=True)

    noOfEle=torch.tensor([B*W*H]) # creating the tensor since recriprocal op supports only tensor

    noOfEle=torch.reciprocal(noOfEle)

    width_mean=torch.mul(width_sum,noOfEle)

    var_sub=torch.sub(x,width_mean) #(x-mean)

    variance=torch.mul(var_sub,var_sub) # (x-mean)*(x-mean)

    var_batch=variance.sum(dim=0,keepdim=True)
    var_height=var_batch.sum(dim=2,keepdim=True)
    var_width=var_height.sum(dim=3,keepdim=True)

    batch_var=torch.mul(var_width,noOfEle) # 1/N*summation((x-mean)*(x-mean)) -> variance

    denominator=torch.add(batch_var,self.eps)  # variance+eps

    sq=torch.sqrt(denominator) #sqrt(denominator)

    rec_sq=torch.reciprocal(sq)

    x_norm=torch.mul(var_sub,rec_sq) # (x-mean)*(1/sq)

    if self.affine:
      self.gamma = self.gamma.reshape(1, C, 1, 1)
      self.beta = self.beta.reshape(1, C, 1, 1)

      # Uses of  gamma and beta
      # Since they are learnable parameters which means,they are updated along with the network weights and biases during training.
      # Beta acts as an offset factor, while gamma acts as a scaling factor.
      # gamma and beta give the network flexibility to undo the normalization if it helps optimize the loss function.
      # So gamma and beta allow the network to scale and shift the normalized values to restore the original distribution if needed to minimize the loss.
      # For ref:https://community.deeplearning.ai/t/batch-normalization-questions/444508/2

      gammaResult=torch.mul(x_norm,self.gamma)
      x_norm=torch.add(gammaResult,self.beta)

    return x_norm


for i in range(5):
    b=torch.randint(100,size=(1,))
    ch=torch.randint(100,size=(1,))
    w=torch.randint(100,size=(1,))
    h=torch.randint(100,size=(1,))
    ip=torch.randn(b,ch,w,h)

    # My decompostion doesn't include the training part.
    cus=CustomBatchNorm2d(ch,affine=False)
    res=cus(ip)
    batch_norm = nn.BatchNorm2d(ch, affine=False)
    print(torch.allclose(batch_norm(ip),res))