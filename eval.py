
import torch
import nltk

def evaluate(sentence, emb_s0,emb_sp, highway, encoder1, generator1, generator2, BATCH_SIZE, device):
    normal_sampler = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])) #for the reparametrisation tricks 
    s_0 = highway(emb_s0(sentence))
    state_so, mu_s0, logvar_s0 = encoder1(s_0)
    std_s0 = torch.exp(logvar_s0/2)                                              
    norm_s0 = normal_sampler.sample(sample_shape=(std_s0.size(0),1)).view(-1,1).to(device)  
    z2 = torch.add(mu_s0,torch.mul(std_s0 ,norm_s0))    
    state_g1 = generator1(s_0)
    world_z2, result_z2, state_z2 = generator2(s_0[:,0], z2, state_g1)
    sentence_with_z2 = world_z2.view(BATCH_SIZE,1,-1)
    world_z2 = highway(emb_sp(world_z2.permute(1,0))).view(BATCH_SIZE,-1)   # next imput # next imput     
    for i in range(1, len(s_0[0]) +10): 
            world_z2, result, state_z2 = generator2(world_z2, z2, state_z2)
            result_z2 = torch.cat((result_z2, result), dim = 1)
            sentence_with_z2 = torch.cat((sentence_with_z2, world_z2.view(BATCH_SIZE,1,-1)), dim = 1)
            world_z2 = highway(emb_sp(world_z2.permute(1,0))).view(BATCH_SIZE,-1)   # next imput
    return sentence_with_z2



def evaluate_blue(iterator, emb_s0, emb_sp, highway, encoder1, generator1, generator2, batch_size, SRC, TRG):
    BLEUscore = 0
    for k, batch in enumerate(iterator):
        e = evaluate(batch.src, emb_s0, emb_sp, highway, encoder1, generator1, generator2)
        for s in range(30):
            sent = ' '
            l_h = []
            for i in range(1,batch_size):
                if TRG.vocab.itos[e[s,i,0]] == "<eos>" or TRG.vocab.itos[e[s,i,0]] == "<pad>" :
                    break
                sent += TRG.vocab.itos[e[s,i,0]]
                sent += ' '
                i +=1
                l_h.append(TRG.vocab.itos[e[s,i,0]])
                l_r1 = []
            for i in range(1,len(batch.src[:,0])):
                if SRC.vocab.itos[batch.src[i,s]] == "<eos>":
                    break
                sent += SRC.vocab.itos[batch.src[i,s]]
                sent += ' '
                l_r1.append(SRC.vocab.itos[batch.src[i,s]])
            l_r2 = []
            for i in range(1,len(batch.trg[:,0])):
                if  TRG.vocab.itos[batch.trg[i,1]] == "<eos>":
                    break
                sent += TRG.vocab.itos[batch.trg[i,s]]
                sent += ' '
                l_r2.append(TRG.vocab.itos[batch.trg[i,s]])
            BLEUscore += nltk.translate.bleu_score.sentence_bleu([ l_r2,  l_r1 ], l_h )
    print(BLEUscore)
    return  BLEUscore /(k * batch_size)