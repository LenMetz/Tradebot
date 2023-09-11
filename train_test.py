import numpy as np
import torch
from utils.loss_functions import CustomLoss, SignWeightedLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(epoch, model, lr, trainloader, testloader, seq_length, batch_size, loss_func, volb = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = loss_func
    print("pretrain test:")
    test(model, testloader, seq_length, batch_size, volb)
    acc = test_conf(model, testloader, seq_length, batch_size, volb)
    for t in range(epoch):
        epoch_loss = 0
        batch_loss = 0
        model.eval()
        for batch_idx, (src, vol, tgt) in enumerate(trainloader):
            
            optimizer.zero_grad()
            # random variable drawn from standard normal (reparametrization trick)
            eps = torch.normal(torch.zeros(src.size()[0]),torch.ones(src.size()[0]))
            src, vol, tgt, eps = src.to(device), vol.to(device), tgt.to(device), eps.to(device)
            if volb:
                prediction, mus, stds = model(src, vol, eps)
            else:
                prediction, mus, stds = model(src, eps)
            #print("pred", prediction.size(), "tgt", tgt.size())
            loss = criterion(prediction, tgt)
            loss = loss.sum()
            #print(loss)
            loss.backward()
            '''for p in model.parameters():
                if p.requires_grad:
                     print(p.grad)'''
            epoch_loss += loss.sum()
            batch_loss += loss.sum()
            optimizer.step()
            #if batch_idx%1000==999:
                #print("batch: ",batch_idx, " -- loss: ",batch_loss/10)
                #batch_loss = 0
                #test(model, testloader, seq_length, batch_size)
        # calculates total gradient of last batch 
        '''grad_sum = 0
        for p in model.parameters():
            if p.requires_grad:
                 grad_sum += p.data.detach().sum()
        print("gradient sun: ", grad_sum)'''
        print("\n\nepoch: ",t, " -- train loss: ",epoch_loss/((batch_idx+1)*batch_size))
        test(model, testloader, seq_length, batch_size, volb)
        acc = test_conf(model, testloader, seq_length, batch_size, volb)
    #print(criterion(src[-1,:,:],tgt).item())
    
    
def test(model, testloader, seq_length, batch_size, volb=False):
    loss_sum = 0
    prediction_list = []
    target_list = []
    target_sum = 0
    model.eval()
    criterion = torch.nn.L1Loss(reduce=False, reduction='sum')
    for batch_idx, (src, vol, tgt) in enumerate(testloader):
        eps = torch.normal(torch.zeros(src.size()[0]),torch.ones(src.size()[0]))
        src, vol, tgt, eps = src.to(device), vol.to(device), tgt.to(device), eps.to(device)
        '''src_last = src[:,-1]
        src_last = torch.unsqueeze(src_last, dim=1).repeat(1,seq_length)
        src = torch.sub(src, src_last)
        tgt = tgt - src_last[:,0]'''
        with torch.no_grad():
            prediction, mus, stds = model(src, vol, eps)
        loss = criterion(prediction, tgt)
        loss_sum += torch.sum(torch.abs(loss.detach()))
        target_sum += torch.sum(torch.abs(tgt))
        prediction_list.extend(prediction.tolist())
        target_list.extend(tgt.tolist())
        #print(loss_sum, target_sum)
    print("pred: \n",prediction[:5], "\ntgt: \n", tgt[:5], "\nmus: \n", mus[:5], "\nstds: \n", stds[:5], "\neps: \n", eps[:5])
    print("Predictions: ", loss_sum/((batch_idx+1)*batch_size), "\nZero Guess: ", target_sum/((batch_idx+1)*batch_size))
    print("Average guess (check if network predicts based on general change): ", torch.sum(torch.tensor(prediction_list))/((batch_idx+1)*batch_size))
    return prediction_list, target_list


def test_conf(model, testloader, seq_length, batch_size, volb=False):
    model.eval()
    tp, tn, fp, fn, mus_t, mus_f, std_t, std_f = 0, 0, 0, 0, 0, 0, 0, 0
    for batch_idx, (src, vol, tgt) in enumerate(testloader):
        eps = torch.normal(torch.zeros(src.size()[0]),torch.ones(src.size()[0]))
        src, vol, tgt, eps = src.to(device), vol.to(device), tgt.to(device), eps.to(device)
        '''src_last = src[:,-1]
        src_last = torch.unsqueeze(src_last, dim=1).repeat(1,seq_length)
        src = torch.sub(src, src_last)
        tgt = tgt - src_last[:,0]'''
        with torch.no_grad():
            prediction, mus, stds = model(src, vol, eps)
        mu_signs = 0.5+0.5*torch.sign(mus)
        tgt_signs = 0.5+0.5*torch.sign(tgt)
        tp += torch.sum(mu_signs*tgt_signs*torch.sign(tgt)*torch.sign(tgt))
        fp += torch.sum(mu_signs*(1-tgt_signs)*torch.sign(tgt)*torch.sign(tgt))
        tn += torch.sum((1-mu_signs)*(1-tgt_signs)*torch.sign(tgt)*torch.sign(tgt))
        fn += torch.sum((1-mu_signs)*tgt_signs*torch.sign(tgt)*torch.sign(tgt))
        mus_t += torch.sum(torch.abs(mus*(mu_signs*tgt_signs+(1-mu_signs)*(1-tgt_signs))))
        mus_f += torch.sum(torch.abs(mus*((1-mu_signs)*tgt_signs+(mu_signs)*(1-tgt_signs))))
        std_t += torch.sum(torch.abs(stds*(mu_signs*tgt_signs+(1-mu_signs)*(1-tgt_signs))))
        std_f += torch.sum(torch.abs(stds*((1-mu_signs)*tgt_signs+(mu_signs)*(1-tgt_signs))))
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    print("tp: ", tp)
    print("fp: ", fp)
    print("tn: ", tn)
    print("fn: ", fn)
    print("accuracy: ", accuracy)
    print("average mu of correct predictions:", mus_t/(tp+tn))
    print("average mu of NOT correct predictions:", mus_f/(fp+fn))
    print("average std of correct predictions:", std_t/(tp+tn))
    print("average std of NOT correct predictions:", std_f/(fp+fn))
    return accuracy
    
def test_conf_ensemble(model1, model2, testloader, seq_length, batch_size, volb=False):
    model1.eval()
    model2.eval()
    tp, tn, fp, fn, mus_t, mus_f, std_t, std_f = 0, 0, 0, 0, 0, 0, 0, 0
    for batch_idx, (src, vol, tgt) in enumerate(testloader):
        eps = torch.normal(torch.zeros(src.size()[0]),torch.ones(src.size()[0]))
        src, vol, tgt, eps = src.to(device), vol.to(device), tgt.to(device), eps.to(device)
        '''src_last = src[:,-1]
        src_last = torch.unsqueeze(src_last, dim=1).repeat(1,seq_length)
        src = torch.sub(src, src_last)
        tgt = tgt - src_last[:,0]'''
        with torch.no_grad():
            prediction1, mus1, stds1 = model1(src, vol, eps)
            prediction2, mus2, stds1 = model2(src, vol, eps)
        mu1_signs = 0.5+0.5*torch.sign(mus1)
        mu2_signs = 0.5+0.5*torch.sign(mus2)
        tgt_signs = 0.5+0.5*torch.sign(tgt)
        tp += torch.sum(mu2_signs*mu1_signs*tgt_signs*torch.sign(tgt)*torch.sign(tgt))
        fp += torch.sum(mu2_signs*mu1_signs*(1-tgt_signs)*torch.sign(tgt)*torch.sign(tgt))
        tn += torch.sum((1-mu2_signs)*(1-mu1_signs)*(1-tgt_signs)*torch.sign(tgt)*torch.sign(tgt))
        fn += torch.sum((1-mu2_signs)*(1-mu1_signs)*tgt_signs*torch.sign(tgt)*torch.sign(tgt))
        mus_t += torch.sum(torch.abs(mus1*(mu1_signs*tgt_signs+(1-mu1_signs)*(1-tgt_signs))))
        mus_f += torch.sum(torch.abs(mus1*((1-mu1_signs)*tgt_signs+(mu1_signs)*(1-tgt_signs))))
        std_t += torch.sum(torch.abs(stds1*(mu1_signs*tgt_signs+(1-mu1_signs)*(1-tgt_signs))))
        std_f += torch.sum(torch.abs(stds1*((1-mu1_signs)*tgt_signs+(mu1_signs)*(1-tgt_signs))))
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    print("tp: ", tp)
    print("fp: ", fp)
    print("tn: ", tn)
    print("fn: ", fn)
    print("accuracy: ", accuracy)
    print("average mu of correct predictions:", mus_t/(tp+tn))
    print("average mu of NOT correct predictions:", mus_f/(fp+fn))
    print("average std of correct predictions:", std_t/(tp+tn))
    print("average std of NOT correct predictions:", std_f/(fp+fn))
    return accuracy

def test_conf_val(model, testloader, seq_length, batch_size, clip):
    if batch_size>1:
        print("BATCHSIZE HAS TO BE 0 FOR THIS TEST!")
        return
    model.eval()
    true_values = []
    true_stds = []
    false_values = []
    false_stds = []
    for batch_idx, (src, tgt) in enumerate(testloader):
        eps = torch.normal(torch.zeros(src.size()[0]),torch.ones(src.size()[0]))
        src, tgt, eps = src.to(device), tgt.to(device), eps.to(device)
        with torch.no_grad():
            prediction, mus, stds = model(src, eps)
        if torch.abs(mus)>=clip:
            if mus*tgt>0:
                true_values.append(torch.abs(mus))
                true_stds.append(torch.abs(stds))
            else:
                false_values.append(torch.abs(mus))
                false_stds.append(torch.abs(stds))
    print("Average absolute values of correctly signed prediction:", torch.mean(torch.tensor(true_values)), 
          "+-", torch.std(torch.tensor(true_values)))
    print("Average absolute values of NOT correctly signed prediction:", torch.mean(torch.tensor(false_values)), 
          "+-", torch.std(torch.tensor(false_values)))
    print("Average absolute stds of correctly signed prediction:", torch.mean(torch.tensor(true_stds)))
    print("Average absolute stds of NOT correctly signed prediction:", torch.mean(torch.tensor(false_stds)))
    

def test_conf_time(model, testloader, seq_length, batch_size, avg_win):
    if batch_size>1:
        print("BATCHSIZE HAS TO BE 0 FOR THIS TEST!")
        return
    model.eval()
    accs = []
    tgts = []
    for batch_idx, (src, tgt) in enumerate(testloader):
        eps = torch.normal(torch.zeros(src.size()[0]),torch.ones(src.size()[0]))
        src, tgt, eps = src.to(device), tgt.to(device), eps.to(device)
        with torch.no_grad():
            prediction, mus, stds = model(src, eps)
        if tgt!=0 and mus!=0:
            musign = 0.5+0.5*torch.sign(mus)
            tgtsign = 0.5+0.5*torch.sign(tgt)
            accs.append(musign*tgtsign + (1-musign)*(1-tgtsign))
            tgts.append(tgt)
    print(torch.mean(torch.tensor(accs)))
    accs_avg = []
    for i in range(len(accs)-avg_win):
        accs_avg.append(torch.mean(torch.tensor(accs[i:i+avg_win])))
    return accs_avg, tgts[:-avg_win]

def test_mus_stds_tgts(model, testloader, seq_length, batch_size):
    model.eval()
    mus = None
    stds = None
    tgts = None
    for batch_idx, (src, vol, tgt) in enumerate(testloader):
        eps = torch.normal(torch.zeros(src.size()[0]),torch.ones(src.size()[0]))
        src, vol, tgt, eps = src.to(device), vol.to(device), tgt.to(device), eps.to(device)
        with torch.no_grad():
            prediction, mu, std = model(src, vol, eps)
        if mus is None:
            mus = mu
            stds = std
            tgts = tgt
        else:
            mus = torch.cat((mus, mu))
            stds = torch.cat((stds, std))
            tgts = torch.cat((tgts, tgt))
    return mus, stds, tgts