import torch
from torch import nn

# for Variable no of Inputs

class CustomBroadcastTensors(nn.Module):
  def __init__(self):
      super().__init__()
  
  def forward(self,*tensors):

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
    print(padded_shapes)

    final_shape = []

    # It will get the final shape
    for i in range(max_rank):
        dim_sizes =[]
        for size in padded_shapes:
          dim_sizes.append(size[i]) # Taking the elementwise size
        final_shape.append(max(dim_sizes))

    final_output=[]
    for i in range(len(tensors)):
      if list(tensors[i].shape)==final_shape:# To avoid the reshaping and tiling for the same broadcasted shape
        final_output.append(tensors[i])
      else:
        tile_shape=[]
        reshaped_tensor=tensors[i].reshape(padded_shapes[i])
        inpShape=list(reshaped_tensor.shape)
        for j in range(len(final_shape)):
          if tile_shape[j]==1:
            tile_shape.append(final_shape[j])
          else:
            tile_shape.append(1)
        final_output.append(torch.tile(reshaped_tensor,tile_shape))

    return final_output
    

class BroadcastTensors(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,*tensors):
    return torch.broadcast_tensors(*tensors)



a = torch.randn(1,6,5,1)
b = torch.randn(1,1,5,7)
c = torch.randn(1,1,1,1)
d = torch.randn(4,1,1,1)
custom_broadcast_tensors=CustomBroadcastTensors()
broadcast_tensors=BroadcastTensors()

custom_results = custom_broadcast_tensors.forward(a, b, c, d)
custom_shapes = [t.shape for t in custom_results]

torch_results = broadcast_tensors.forward(a, b, c, d)
torch_shapes = [t.shape for t in torch_results]


for cus_t, torch_t in zip(custom_results, torch_results):
    assert torch.allclose(cus_t, torch_t)