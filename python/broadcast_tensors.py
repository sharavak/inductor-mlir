import torch
from torch import nn

# for Variable no of Inputs
class CustomBroadcastTensors(nn.Module):
  def __init__(self):
      super().__init__()
  
  def forward(self,*tensors):
    for tensor in tensors:
      if isinstance(tensor, (int, float)):  # Check if input is a raw scalar (int or float)
        raise RuntimeError("Broadcast shape is not possible for raw scalar input")
        

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

    # It will get the final shape
    for i in range(max_rank):
        dim_sizes =[]
        for size in padded_shapes:
          dim_sizes.append(size[i]) # Taking the elementwise size
        final_shape.append(max(dim_sizes))

    
    final_shape_tensor=torch.tensor(final_shape)
    for i in range(len(tensors)):
        pad_shape_tensor=torch.tensor(padded_shapes[i])
        condition=torch.logical_or(torch.eq(pad_shape_tensor,final_shape_tensor),torch.eq(pad_shape_tensor,1))
        if not torch.all(condition):
          raise RuntimeError("Broadcast shape is not possible")
       
        
    final_output=[]
    for i in range(len(tensors)):
      if list(tensors[i].shape)==final_shape:# To avoid the reshaping and tiling for the same broadcasted shape
        final_output.append(tensors[i])
      else:
        tile_shape=[]
        reshaped_tensor=tensors[i].reshape(padded_shapes[i])
        if list(reshaped_tensor.shape)==final_shape:
          final_output.append(reshaped_tensor)
          continue
  
        inpShape=list(reshaped_tensor.shape)
        for j in range(len(final_shape)):
          if inpShape[j]==1:
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


def test_cases():
    cases = [
        (torch.randn(3,), torch.randn(1, 3), torch.randn(1, 1, 3)),
        (torch.randn(2, 3), torch.randn(1, 3, 1), torch.randn(1, 1, 3)),
        (torch.randn(2, 1, 5), torch.randn(1, 3, 1), torch.randn(3, 1, 5)),
        ((2,3)),

        (torch.tensor(5.0), torch.randn(4, 3)),
        (torch.tensor(1.0), torch.tensor(2.0), torch.randn(3, 4)), 

        (torch.randn(2, 3, 4), torch.ones(1, 1, 1)),  
        (torch.randn(1, 4, 1), torch.ones(4, 1, 3)),  
        (torch.randn(1, 1, 3), torch.ones( 1, 3)),  

        (torch.randn(0, 3), torch.randn(0, 3, 5)),  
        (torch.randn(1, 0, 4), torch.randn(3, 0, 1)),  
        (torch.randn(1, 3, 4), torch.randn(0, 3, 1)), 
        (torch.randn(1,3,5),torch.randn(2,2,1)),
        (torch.randn(3, 5), torch.randn(2, 4)),  
        (torch.randn(2, 3), torch.randn(3, 2)),  
        
        (torch.randn(1, 3, 1), torch.randn(3, 1, 4), torch.randn(1, 3, 4)), 
        (torch.randn(1, 4, 5), torch.randn(1, 1, 5), torch.randn(4, 1, 5), torch.randn(4, 5))
    ]

    custom_broadcast_tensors = CustomBroadcastTensors()
    broadcast_tensors = BroadcastTensors()
    
    for tensors in cases:
        try:
            custom_results = custom_broadcast_tensors.forward(*tensors)
            torch_results = broadcast_tensors.forward(*tensors)
            for cus_t, torch_t in zip(custom_results, torch_results):
                assert torch.allclose(cus_t, torch_t)
        except Exception as e:
            print(f"RuntimeError: {e} for the tensor",)

test_cases()