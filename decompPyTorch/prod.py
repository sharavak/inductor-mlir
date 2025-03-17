import torch

inp=torch.tensor([
    [
      [1,2,3,4],
      [4,5,6,5],
      [6,7,8,2]
      ],[
      [1,2,3,4],
      [4,5,6,5],
      [6,7,8,2]
      ]
    ])

def customProd(x,dim=0,keepdim=False):
  # dim can be int or list of tuples
  
  newShape=[]
  if isinstance(dim,tuple):
    if keepdim:
      for i in dim:
        x=torch.prod(x,dim=i,keepdim=True)
      return x
    else:
      for i in range(len(x.shape)):
        if i not in dim:
          newShape.append(x.shape[i])
     
      for i in dim:
        x=torch.prod(x,dim=i,keepdim=True)
      return x.reshape(newShape)

  elif not keepdim:# if the keepdim is false
    size=len(x.shape)
    # reducing the shape according to the dim
    # For eg -> if shape=[2,3,4] and  dim=1 -> then resultant shape is [2,4]
    for i in range(size):
      if i!=dim:
        newShape.append(x.shape[i])
    red_prod=torch.prod(x,dim=dim,keepdim=True) # output shape is (x.shape)
    return red_prod.reshape(newShape) # reshape is used to get the new shape
  else:
    # if keepdim is true, then the specified dim is reduced to 1
    # For eg -> shape=[5,6,9],dim=2 then -> output is [5,6,1]
    red_prod=torch.prod(x,dim=dim,keepdim=keepdim)#
    return red_prod

print(torch.allclose(customProd(inp,dim=2,keepdim=True),torch.prod(inp,dim=2,keepdim=True)))
print(torch.allclose(customProd(inp,dim=2,keepdim=False),torch.prod(inp,dim=2,keepdim=False)))

# For dim with list of tuples

dim=[0,1]
res=torch.clone(inp)
for i in dim:
  inp=torch.prod(inp,dim=i,keepdim=True)
print(torch.allclose(inp,customProd(inp,dim=dim,keepdim=True)))