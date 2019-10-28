import torch
import torch.nn as nn

from torch import optim

import time

def train(num_epoch, train_iterator, emb_s0, emb_sp, highway, encoder1, encoder2, generator1, generator2, discriminator, device, pad_idx):
    
    def PG_discriminator_loss(trg, output, reward, BATCH_SIZE):
        #Inspired by the example in http://karpathy.github.io/2016/05/31/rl/ and #https://github.com/suragnair/seqGAN/blob/master/generator.py
        """trg : [seq len, batch size]
            output : [seq _len, batch _size, vocab dim]
            reward : [batch size,1]"""
        loss = 0
        for i in range(len(trg) - 1):
            for batch in range(BATCH_SIZE):
                loss += -output[i][batch][trg[i][batch]] * reward[batch]
        return loss / BATCH_SIZE

    BATCH_SIZE = train_iterator.batch_size
    PAD_IDX = pad_idx 
    criterion_rec = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    
    highway.train()
    optimizer_highway = optim.Adam(highway.parameters(), lr=0.0001)
   
    encoder1.train()
    optimizer_encodeur1 = optim.Adam(encoder1.parameters(), lr=0.0001)
   
    encoder2.train()
    optimizer_encodeur2 = optim.Adam(encoder2.parameters(), lr=0.0001)
    
    generator1.train()
    optimizer_generator1 = optim.Adam(generator1.parameters(), lr=0.0001)
    
    generator2.train()
    optimizer_generator2 = optim.Adam(generator2.parameters(), lr=0.0001)
   
    discriminator.train() 
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0001)

    normal_sampler = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0])) #for the reparametrisation tricks
    
    lambda2 = 0.5
    lambda1 = 1 - lambda2
    
    lambda3 = 0.01
    
    list_loss_E1 = []
    list_loss_E2 = []
    list_loss_G = []
    list_loss_D = []
    
    t = time.time()
    for epoch in range(num_epoch):

          for k, batch in enumerate(train_iterator):
            if len(batch) != BATCH_SIZE:
              print("end of the iterator")
              break

            ######################
            # embedding using glove d300
            ######################

            trg = batch.trg # a passer en question 1
            #s_0 = emb_s0(batch.question1) #highway for preprocessing
            #s_p = emb_sp(batch.question2)
            s_0 = highway(emb_s0(batch.src)) #highway for preprocessing
            s_p = highway(emb_sp(batch.trg))

            #################################
            # encodeur  part
            ####################################

            state_so, mu_s0, logvar_s0 = encoder1(s_0)
            state_s0_sp, mu_s0_sp, logvar_s0_sp = encoder2(s_0, state_so)

          #https://github.com/pytorch/examples/blob/master/vae/main.py
                #https://arxiv.org/pdf/1312.6114.pdf appendix b
            L_KLD = -0.5 * torch.sum(1 + logvar_s0_sp - mu_s0_sp.pow(2) - logvar_s0_sp.exp())      
          
            ################
            # reparametrization tricks
            ################
            std_s0 = torch.exp(logvar_s0/2)
            std_s0_sp = torch.exp(logvar_s0_sp/2)                                               
            norm_s0 = normal_sampler.sample(sample_shape=(std_s0.size(0),1)).view(-1,1).to(device)
            norm_s0_sp = normal_sampler.sample(sample_shape=(std_s0_sp.size(0),1)).view(-1,1).to(device)
            z1 = torch.add(mu_s0_sp,torch.mul(std_s0_sp ,norm_s0_sp))
            z2 = torch.add(mu_s0,torch.mul(std_s0 ,norm_s0))

            ########################
            # generator part
            ########################

            state_g1 = generator1(s_0)

            world_z1, result_z1, state_z1 = generator2(s_0[:,0], z1, state_g1) 
            world_z2, result_z2, state_z2 = generator2(s_0[:,0], z2, state_g1)
            sentence_with_z1 = world_z1.view(BATCH_SIZE,1,-1)
            sentence_with_z2 = world_z2.view(BATCH_SIZE,1,-1)

            world_z1 = highway(emb_sp(world_z1.permute(1,0))).view(BATCH_SIZE,-1)  # next imput
            world_z2 = highway(emb_sp(world_z2.permute(1,0))).view(BATCH_SIZE,-1)   # next imput


            for i in range(1, len(s_p[0]) -1):
                world_z1, result, state_z1 = generator2(world_z1, z1, state_z1)  # #wolrd: [batchsize, 1] ,result : [batch size, 1, vocab sp dim]
                result_z1 = torch.cat((result_z1, result), dim = 1)
                sentence_with_z1 = torch.cat((sentence_with_z1, world_z1.view(BATCH_SIZE,1,-1)), dim = 1)
                world_z1 = highway(emb_sp(world_z1.permute(1,0))).view(BATCH_SIZE,-1)   #embbeding next imput
                #world_z1 = emb_sp(world_z1.permute(1,0)).view(BATCH_SIZE,-1)

                world_z2, result, state_z2 = generator2(world_z2, z2, state_z2)
                result_z2 = torch.cat((result_z2, result), dim = 1)
                sentence_with_z2 = torch.cat((sentence_with_z2, world_z2.view(BATCH_SIZE,1,-1)), dim = 1)
                world_z2 = highway(emb_sp(world_z2.permute(1,0))).view(BATCH_SIZE,-1)   # next imput
                #world_z2 = emb_sp(world_z2.permute(1,0)).view(BATCH_SIZE,-1) 
                #print("sentence_with_z2 %s"  % str(sentence_with_z2.size()))

            #sentence_with_z2 =  [batch size, sentence len, 1]
            #sentence_with_z1 = [batch size, sentence len, ]


            result_z2 = result_z2.transpose(1,0)
            result_z1 = result_z1.transpose(1,0)

            trg_for_Lrec = trg[1:,].view(-1)
            result_z2_for_Lrec = result_z2.contiguous().view(-1, result_z2.shape[-1]) 

            result_z1_for_Lrec = result_z1.contiguous().view(-1, result_z1.shape[-1])
                        

            Lrec1 = criterion_rec(result_z1_for_Lrec, trg_for_Lrec)
            Lrec2 = criterion_rec(result_z2_for_Lrec, trg_for_Lrec)


            ########################################
            #                 discriminator part
            ########################################

            imp = highway(emb_sp(sentence_with_z2.transpose(1,0).reshape(-1, BATCH_SIZE)))
            #imp = emb_sp(sentence_with_z2.transpose(1,0).contiguous().view(-1, BATCH_SIZE))

            d_s2 = discriminator(imp)
            d_sp = discriminator(s_p)

            Loss_D = -torch.log(torch.mean(d_sp))  -torch.log(1 - torch.mean(d_s2))
            loss_DG = PG_discriminator_loss(trg, result_z2, d_sp, BATCH_SIZE) #dsp or ds2 
           
            ########################################
           # loss_computation and backward part
           ########################################
            if (k*(epoch+1)) <= 10000 :#warm up trick
                lambda2 = 0.5 * ((k*(epoch+1))/10000)
                lambda1 = 1 - lambda2

                lambda3 = 0.01 * ((k*(epoch+1))/10000)

            optimizer_encodeur1.zero_grad()
            loss_E1 = lambda1 * (L_KLD + Lrec1) + lambda2 * Lrec2 + lambda3 * loss_DG   #=∇E1(λ1LKL+λ1Lrec1+λ2Lrec2+λ3LDG)
            loss_E1.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(encoder1.parameters(), 10, norm_type=2)
            optimizer_encodeur1.step()
            
            optimizer_highway.zero_grad()
            optimizer_encodeur1.zero_grad()
            loss_E1 = lambda1 * (L_KLD + Lrec1) + lambda2 * Lrec2 + lambda3 * loss_DG   #=∇E1(λ1LKL+λ1Lrec1+λ2Lrec2+λ3LDG)
            loss_E1.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(encoder1.parameters(), 10, norm_type=2)
            torch.nn.utils.clip_grad_norm_(highway.parameters(), 10, norm_type=2)
            optimizer_encodeur1.step()
            optimizer_highway.step()


     
            optimizer_encodeur2.zero_grad()
            loss_E2 = (L_KLD + Lrec1)   #∇E2(LKL+Lrec1);
            loss_E2.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(encoder2.parameters(), 10, norm_type=2)
            optimizer_encodeur2.step()

            optimizer_generator2.zero_grad()
            optimizer_generator1.zero_grad()
            loss_g = lambda1 * Lrec1 + lambda2 * Lrec2 + lambda3 * loss_DG   #∇G(λ1Lrec1+λ2Lrec2+λ3LDG);
            loss_g.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(generator2.parameters(), 10, norm_type=2)
            torch.nn.utils.clip_grad_norm_(generator1.parameters(), 10, norm_type=2)       
            optimizer_generator2.step()
            optimizer_generator1.step()


            optimizer_discriminator.zero_grad()
            Loss_D.backward() #∇LD
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5, norm_type=2) 
            optimizer_discriminator.step()



            list_loss_E1.append(float(loss_E1))
            list_loss_E2.append(float(loss_E2))
            list_loss_G.append(float(loss_g))
            list_loss_D.append(float(Loss_D))


            loss_E1 = 0
            loss_E2 = 0
            loss_g = 0
            Loss_D = 0
     

            if k%50==0:
                
                print("k %s" % k)
                print("epoch %s" % epoch)

                print("loss")
                print(time.time() -t)
                print( [list_loss_E1[-1], list_loss_E2[-1], list_loss_G[-1], list_loss_D[-1] ])
         
    return  [list_loss_E1, list_loss_E2, list_loss_G, list_loss_D ], [highway, encoder1, encoder2, generator1, generator2, discriminator]


