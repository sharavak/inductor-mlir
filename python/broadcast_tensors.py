import torch


# for 2 inputs
def custom_broadcast_tensors(a,b):
  rank1,rank2=len(a.shape),len(b.shape)
  ns1=list(a.shape)
  ns2=list(b.shape)
  newShape=[]
  if rank1<rank2:
    for i in range(rank2-rank1): #Initially padding the ones from the left index
      newShape.append(1)
    for i in ns1:
      newShape.append(i)
  else:
    for i in range(rank1-rank2):
      newShape.append(1)
    for i in ns2:
      newShape.append(i)
  if rank1<rank2:
    finalShape=[1]*rank2
    for i in range(rank2-1,-1,-1): # starting from the rightmost trailing end
      finalShape[i]=max(newShape[i],ns2[i])
  else:
    finalShape=[1]*rank1
    for i in range(rank1-1,-1,-1):
      finalShape[i]=max(newShape[i],ns1[i])
  resRank=len(finalShape)

  r1=[]
  for i in range(resRank-rank1):
    r1.append(1)
  for i in ns1:
    r1.append(i)
  r2=[]
  for i in range(resRank-rank2):
    r2.append(1)
  for i in ns2:
    r2.append(i)

  a=a.reshape(r1)
  b=b.reshape(r2)
  t1=[]
  for i in range(resRank):
    if r1[i]==1:
      t1.append(finalShape[i])
    else:
      t1.append(1)
  t2=[]

  for i in range(resRank):
    if r2[i]==1:
      t2.append(finalShape[i])
    else:
      t2.append(1)
  
  r1=torch.tile(a,t1) 
  r2=torch.tile(b,t2)

  return r1,r2


a = torch.randn(1,6,5)                  
b = torch.randn(7,1,1) 
a1,b1=custom_broadcast_tensors(a,b)
t1,t2=torch.broadcast_tensors(a,b)
assert torch.allclose(a1,t1) and torch.allclose(b1,t2)




# for Variable no of Inputs
def custom_broadcast_tensors(*tensors):
    shapes = [list(t.shape) for t in tensors]
    max_rank = max(len(s) for s in shapes)  # for getting the max_rank

    padded_shapes = []
    for s in shapes:
        padded=[]
        # this will pad the ones from the leftmost part
        for i in range(max_rank - len(s)):
          padded.append(1)
        for i in s:
          padded.append(i)   
        padded_shapes.append(padded)
    
    
    final_shape = [] 

    # It will get the final broadcasted shape
    for i in range(max_rank):
        dim_sizes =[]
        for size in padded_shapes:
          dim_sizes.append(size[i]) # Taking the elementwise size
        final_shape.append(max(dim_sizes))

    reshaped_tensors=[]
    for i in range(len(tensors)):
      reshaped_tensors.append(tensors[i].reshape(padded_shapes[i]))

    final_output=[]

    for i in range(len(tensors)):
      t1=[]
      s1=list(reshaped_tensors[i].shape)
      for j in range(len(final_shape)):
        if s1[j]==1:
          t1.append(final_shape[j])
        else:
          t1.append(1)
      final_output.append(torch.tile(reshaped_tensors[i],t1))

    return final_output


a = torch.randn(3)                  
b = torch.randn(1, 4, 3, 1)         
c = torch.randn(1, 3)               
d = torch.randn(4, 1, 3)            

custom_results = custom_broadcast_tensors(a, b, c, d)
custom_shapes = [t.shape for t in custom_results]

torch_results = torch.broadcast_tensors(a, b, c, d)
torch_shapes = [t.shape for t in torch_results]


for cus_t, torch_t in zip(custom_results, torch_results):
    assert torch.allclose(cus_t, torch_t)