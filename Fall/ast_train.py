import os
import datetime
from stats import *
from utils import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast, GradScaler

def train(audio_model, train_loader, test_loader, lr=0.001, n_epochs=10):
    device = torch.device("cuda:2")
    torch.set_grad_enabled(True)
    
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    
    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time()-start_time])
        
        with open("progress.pkl", "wb") as f:
            pickle.dump(progress, f)
            
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model, [2,1],2)
        
    audio_model = audio_model.to(device)
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, lr, weight_decay=5e-7, betas=(0.95, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
    main_metrics = 'acc'
    loss_fn = nn.CrossEntropyLoss()
    warmup = False
    print('now training with main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(main_metrics), str(loss_fn), str(scheduler)))
    
    epoch += 1
    scaler = GradScaler()
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([n_epochs, 10])
    audio_model.train()
    while epoch < n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        
        for i, d in enumerate(train_loader):
            audio_input = d['input']
            labels = d['label']
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            data_time.update(time.time(), - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()
            
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * lr
                for param_groups in optimizer.param_groups:
                    param_groups['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))
                
            with autocast():
                audio_output = audio_model(audio_input)
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(audio_output, labels)
                else:
                    loss = loss_fn(audio_output, labels)
                    
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])
            
            print_step = global_step % 50 == 0
            early_print_step = epoch==0 and global_step % (50/10) == 0
            print_step = print_step or early_print_step
            
            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Total Time {per_sample_time.avg:.5f}\t'
                  'Data Time {per_sample_data_time.avg:.5f}\t'
                  'DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return 
            end_time = time.time()
            global_step += 1
            
        print('start validation')
        
        stats, valid_loss = validate(audio_model, test_loader, epoch=10)
        
        #mAP = np.mean([stat['AP'] for stat in stats])
        #mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']
        
        #middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        #middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        #average_precision = np.mean(middle_ps)
        #average_recall = np.mean(middle_rs)
        
        print("acc: {:.6f}".format(acc))
        #print("AUC: {:.6f}".format(mAUC))
        #print("Avg Precision: {:.6f}".format(average_precision))
        #print("Avg Recall: {:.6f}".format(average_recall))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))
        
        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch
        
        
        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        _save_progress()
        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()        

        #mAP = np.mean([stat['AP'] for stat in stats])
        #mAUC = np.mean([stat['auc'] for stat in stats])
        #middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        #middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        #average_precision = np.mean(middle_ps)
        #average_recall = np.mean(middle_rs)
        #wa_result = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC)]
        print('---------------Training Finished---------------')
        print('weighted averaged model results')
        #print("mAP: {:.6f}".format(mAP))
        #print("AUC: {:.6f}".format(mAUC))
        #print("Avg Precision: {:.6f}".format(average_precision))
        #print("Avg Recall: {:.6f}".format(average_recall))
        #print("d_prime: {:.6f}".format(d_prime(mAUC))) 
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))
        
def validate(audio_model, val_loader, epoch=10):
    device = torch.device("cuda:2")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model, [2,1],2)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, d in enumerate(val_loader):
            audio_input = d['input']
            labels = d['label']
            audio_input = audio_input.to(device)

            audio_output = audio_model(audio_input)
            #_, predictions = audio_output.max(1)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()
        
            A_predictions.append(predictions)
            A_targets.append(labels)

            labels = labels.to(device, dtype=torch.int64)
            #audio_output = audio_output.to('cpu').detach()
            loss = loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        
        stats = calculate_stats(audio_output, target)
    return stats, loss

                