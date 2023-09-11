import torch

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion = torch.nn.L1Loss(reduce = False)

    def forward(self, output, target):
        loss = self.criterion(output, target)
        sg = torch.nn.Tanh()
        regu = (1/(sg(torch.abs(output)/10)+0.0001))
        #print("loss before: ", loss)
        #print("scaling: ", regu)
        loss = regu+loss
        #print("loss after:" , loss)
        return loss
    
class SignWeightedLoss(torch.nn.Module):
    def __init__(self, weight):
        super(SignWeightedLoss, self).__init__()
        self.criterion = torch.nn.L1Loss(reduce = False)
        self.weight = weight
        
    def forward(self, output, target):
        loss = self.criterion(output, target)
        signs = output*target
        reg = torch.zeros(signs.size()).to(loss.device)
        reg[torch.where(signs<0)] = 1
        return loss + self.weight*reg

class SignWeightedTgtLoss(torch.nn.Module):
    def __init__(self, weight):
        super(SignWeightedTgtLoss, self).__init__()
        self.criterion = torch.nn.L1Loss(reduce = False)
        self.weight = weight
        
    def forward(self, output, target):
        loss = self.criterion(output, target)
        signs = output*target
        regInd = torch.zeros(signs.size()).to(loss.device)
        regInd[torch.where(signs<0)] = 1
        return loss + self.weight*regInd*torch.abs(target)




