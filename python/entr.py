#torch.special.entr
import torch
def customEntr(input):
    # Formula -> -x*ln(x)

    # This Op performs operation in elementwise

    # mask is a boolean tensor
    
    pos_mask = torch.gt(input ,0) #filtering the postive mask

    neg_mask=torch.lt(input ,0)

    neg_input=torch.mul(-1,input[pos_mask]) # multiplying with the input to get the negative input

    output=torch.log(input[pos_mask]) # taking the log

    input[pos_mask] = torch.mul(neg_input , output) # -x*log(x)

    input[neg_mask] = float('-inf')  # replacing with -inf

    zero_mask=torch.eq(input,0)

    input[zero_mask] = 0 # Replacing with  zero

    return input
inp=torch.tensor([-1,-1,-1,-1,-1],dtype=torch.float32)
assert torch.allclose(customEntr(inp),torch.special.entr(inp))

inp=torch.tensor([2,3,5,8,9],dtype=torch.float32)
assert torch.allclose(customEntr(inp),torch.special.entr(inp))

inp=torch.randn(8,5,6,9)
assert torch.allclose(torch.allclose(customEntr(inp),torch.special.entr(inp)))