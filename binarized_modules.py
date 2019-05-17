import torch.nn as nn

def Binarize(tensor,quant_mode='det'):
        return tensor.sign()
    
class BinarizeSign(nn.Module):

    def __init__(self):
        super(BinarizeSign, self).__init__()

    def forward(self, input):

        input.data=Binarize(input.data)
        out = input
        return out

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        out = nn.functional.linear(input, self.weight)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out