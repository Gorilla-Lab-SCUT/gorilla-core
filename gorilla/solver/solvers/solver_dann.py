import torch
from torch.autograd import Variable
import time
import os
import numpy as np

from solver.base_solver import BaseSolver
from utils.utils import save_summary
from utils.utils_train import adjust_learning_rate, accuracy
# TODO: 少了个criterion

class solver_DANN(BaseSolver):
    # TODO: 还没删改
    def __init__(self, model, dataloaders, optimizer, lr_scheduler, cfg):
        super(solver_DANN, self).__init__(model, dataloaders, optimizer, lr_scheduler, cfg)
        
    def solve(self):
        print("Begin training")
        tmp = int(self.epoch) # for deep copy
        self.epoch = -2
        self.test()
        self.epoch = tmp
        print(len(self.dataloaders["train_src"]), len(self.dataloaders["train_tgt"]), len(self.dataloaders["val_tgt"]), len(self.dataloaders["test_tgt"]))
        self.max_drawdown = 0
        while self.epoch < self.cfg.epochs:
            self.train()

            if self.epoch % self.cfg.test_interval == (self.cfg.test_interval - 1):
                counter = self.test()
                current_drawdown = self.AMdict["top1_target_val"].max - self.AMdict["top1_target_val"].val
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
                
                print("counter:", counter)
                if self.cfg.early:
                    if counter >= self.cfg.patience:
                        break
            self.epoch += 1

    def train(self, **kwargs):
        self.model.G_f.train()
        self.model.G_y.train()
        self.model.G_d.train()

        for key in self.optimizers.keys():
            adjust_learning_rate(self.optimizers[key], self.epoch, self.cfg, mode='auto', value=0.1)    
            if self.cfg.debug:
                print("lr of optimizer {}:".format(key))
                for param_group in self.optimizers[key].param_groups:
                    print(param_group['lr'])

        go_on = True
        while go_on:
#             print(f"epoch {self.epoch} iter {self.iter}")
            end = time.time()
            # prepare the data for the model forward and backward
            # note that DANN is used on the condition that the label of target dataset is not available
            source_data, source_gt = self.get_samples('train_src')
            target_data, _ = self.get_samples('train_tgt')

            if torch.cuda.is_available():
                source_data = source_data.cuda(async=True)
                source_gt = source_gt.cuda(async=True)
                target_data = target_data.cuda(async=True)

            self.AMdict["data_time_train_per_epoch"].update(time.time() - end)
            # prepare domain labels
            domain_labels_source = torch.zeros((source_data.size()[0])).type(torch.LongTensor)
            domain_labels_target = torch.ones((target_data.size()[0])).type(torch.LongTensor)
            if torch.cuda.is_available():
                domain_labels_source = domain_labels_source.cuda(async=True)
                domain_labels_target = domain_labels_source.cuda(async=True)

            # compute the output of source domain and target domain
            outputs = self.model(source_data, target_data)

            # compute the category loss of feature_source
            loss_C = self.criterions['CrossEntropy'](outputs_source, source_gt)

            # compute the domain loss of feature_source and target_feature
            p = float(self.epoch) / self.cfg.epochs
            constant = 2. / (1. + np.exp(-self.cfg.gamma * p)) - 1
            print('constant:', constant) 
            preds_source = self.models["G_d"](feature_source, constant)
            preds_target = self.models["G_d"](feature_target, constant)
            domain_loss_source = self.criterions['CrossEntropy'](preds_source, domain_labels_source)
            domain_loss_target = self.criterions['CrossEntropy'](preds_target, domain_labels_target)
            loss_G = domain_loss_target + domain_loss_source
            print('loss_G:', loss_G)

            loss_total = 0
            for key in self.AMdict["loss_weights"].keys():
                loss_total += self.AMdict["loss_weights"][key] * eval("{}".format(key))            
            self.optimizers["G"].zero_grad()
            loss_total.backward()
            self.optimizers["G"].step() 

            pred_acc1_source, pred_acc5_source = accuracy(outputs_source, source_gt, topk=(1, 5))

            # measure accuracy and record loss          
            for key in self.AMdict["loss_weights"].keys():
                self.AMdict["loss"]["{}_train_per_epoch".format(key)].update( eval("{}.data".format(key)) )

            self.AMdict["loss"]["loss_total_train_per_epoch"].update(loss_total.data)
            self.AMdict["top1_source_train_per_epoch"].update(pred_acc1_source)
            self.AMdict["top5_source_train_per_epoch"].update(pred_acc5_source)
            self.AMdict["batch_time_train_per_epoch"].update(time.time() - end)

            if self.iter % self.iters_per_epoch == self.iters_per_epoch - 1:
                # update upper level indicator
                self.AMdict["data_time_train"].update(self.AMdict["data_time_train_per_epoch"].sum)
                for key in self.AMdict["loss_weights"].keys():
                    self.AMdict["loss"]["{}_train".format(key)].update(self.AMdict["loss"]["{}_train_per_epoch".format(key)].avg)
                self.AMdict["loss"]["loss_total_train"].update(self.AMdict["loss"]["loss_total_train_per_epoch"].avg)

                self.AMdict["top1_source_train"].update(self.AMdict["top1_source_train_per_epoch"].avg)
                self.AMdict["top5_source_train"].update(self.AMdict["top5_source_train_per_epoch"].avg)
                self.AMdict["batch_time_train"].update(self.AMdict["batch_time_train_per_epoch"].sum)
                # and reset lower level indicator
                self.AMdict["data_time_train_per_epoch"].reset()
                for key in self.AMdict["loss_weights"].keys():
                    self.AMdict["loss"]["{}_train_per_epoch".format(key)].reset()      
                self.AMdict["loss"]["loss_total_train_per_epoch"].reset()
                self.AMdict["top1_source_train_per_epoch"].reset()
                self.AMdict["top5_source_train_per_epoch"].reset()
                self.AMdict["batch_time_train_per_epoch"].reset()

                if self.epoch % self.cfg.log_interval == self.cfg.log_interval - 1:
                    loss_tmp = self.AMdict["loss"]["loss_total_train"]
                    loss_string = 'total_loss {:.4f} ({:.4f})    '.format(loss_tmp.val, loss_tmp.avg)
                    for key in self.AMdict["loss_weights"].keys():
                        loss_tmp = self.AMdict["loss"]["{}_train".format(key)]
                        loss_string += '{} {:.4f} ({:.4f})    '.format(key, loss_tmp.val, loss_tmp.avg)

                    print_string = ('Tr self.epoch [{0}/{1}]    '
                              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})    '
                              'DT {data_time.val:.3f} ({data_time.avg:.3f})    '
                              'S@1 {top1_source.val:.3f} ({top1_source.avg:.3f})    '
                              'S@5 {top5_source.val:.3f} ({top5_source.avg:.3f})    ').format(
                              self.epoch + 1, self.cfg.epochs, 
                              batch_time=self.AMdict["batch_time_train"], data_time=self.AMdict["data_time_train"],
                              top1_source=self.AMdict["top1_source_train"], top5_source=self.AMdict["top5_source_train"], 
                              ) 

                    print_string += loss_string
                    print(print_string)
                    self.logger.log(print_string + '\n')
                    
                    self.writer.add_scalars('loss', {'total': self.AMdict["loss"]["loss_total_train"].val}, self.epoch + 1)
                    self.writer.add_scalars('acc', {'train': self.AMdict["top1_source_train"].val}, self.epoch + 1)                        
                # save checkpoint every some epochs
                if self.epoch % self.cfg.record_interval == (self.cfg.record_interval - 1):
                    filepath = os.path.join("log", "summary.npy")
                    filename = "latest_model.pth.tar"
                    dir_save_file = os.path.join(self.cfg.log, filename)
                    state = {
                        "epoch": self.epoch + 1,
                        "arch": self.cfg.arch,
                        "loss": loss_total,
                        }
                    for key in self.models.keys():
                        state.update({key: self.models[key].state_dict()})

                    save_summary(filepath, state, dir_save_file, loss_total, desc='loss in self.epoch {}'.format(self.epoch + 1), smaller=True, overwrite=True) 
                                 
                go_on = False
            self.iter += 1

    def test(self, **kwargs):
        for key in self.models.keys():
            self.models[key].eval()     

        end = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.dataloaders["val_tgt"]):
                target_data, target_gt = data
                if torch.cuda.is_available():
                    if torch.__version__ < '1.0.0':
                        target_data = Variable(target_data.cuda(async=True))
                        target_gt = Variable(target_gt.cuda(async=True))
                    else:
                        target_data = target_data.cuda(async=True)
                        target_gt = target_gt.cuda(async=True)

                outputs_target = self.models["G_y"](self.models["G_f"](target_data))

                pred_acc1_target, pred_acc5_target = accuracy(outputs_target, target_gt, topk=(1, 5))
                batch_size_target = target_data.size(0)
                self.AMdict["top1_target_val_per_epoch"].update(pred_acc1_target, batch_size_target)
                self.AMdict["top5_target_val_per_epoch"].update(pred_acc5_target, batch_size_target)
                
            for i, data in enumerate(self.dataloaders["test_tgt"]):
                target_data, target_gt = data
                if torch.cuda.is_available():
                    if torch.__version__ < '1.0.0':
                        target_data = Variable(target_data.cuda(async=True))
                        target_gt = Variable(target_gt.cuda(async=True))
                    else:
                        target_data = target_data.cuda(async=True)
                        target_gt = target_gt.cuda(async=True)

                outputs_target = self.models["G_y"](self.models["G_f"](target_data))

                pred_acc1_target, pred_acc5_target = accuracy(outputs_target, target_gt, topk=(1, 5))
                batch_size_target = target_data.size(0)
                self.AMdict["top1_target_test_per_epoch"].update(pred_acc1_target, batch_size_target)
                self.AMdict["top5_target_test_per_epoch"].update(pred_acc5_target, batch_size_target)

        self.AMdict["total_time_val"].update(time.time() - end)
        self.AMdict["top1_target_val"].update(self.AMdict["top1_target_val_per_epoch"].avg) # the average measure in the whole validation set is the current measure in the AverageMeter
        self.AMdict["top5_target_val"].update(self.AMdict["top5_target_val_per_epoch"].avg)
        self.AMdict["top1_target_test"].update(self.AMdict["top1_target_test_per_epoch"].avg)
        self.AMdict["top5_target_test"].update(self.AMdict["top5_target_test_per_epoch"].avg)        
        self.AMdict["top1_target_val_per_epoch"].reset()
        self.AMdict["top5_target_val_per_epoch"].reset()
        self.AMdict["top1_target_test_per_epoch"].reset()
        self.AMdict["top5_target_test_per_epoch"].reset()

        print_string = ('    Te self.epoch [{0}/{1}]    '
                    'Time {total_time.val:.3f} ({total_time.avg:.3f})    '
                    'T@1 {top1_target.val:.3f} ({top1_target.avg:.3f})    '
                    'T@5 {top5_target.val:.3f} ({top5_target.avg:.3f})    ').format(
                    self.epoch + 1, self.cfg.epochs, total_time=self.AMdict["total_time_val"], 
                    top1_target=self.AMdict["top1_target_val"], top5_target=self.AMdict["top5_target_val"])

        print(print_string)
        self.logger.log(print_string + '\n')

        if self.AMdict["top1_target_val"].epochs_max_not_refresh == 0: # it means that the performance of current self.epoch is the best one
            filepath = os.path.join("log", "summary.npy")
            state = {
                "epoch": self.epoch + 1,
                "arch": self.cfg.arch,
                "best_prec1": self.AMdict["top1_target_val"].val,
                }
            for key in self.models.keys():
                state.update({key: self.models[key].state_dict()})
            filename = "best_model.pth.tar"
            dir_save_file = os.path.join(self.cfg.log, filename) 
            save_summary(filepath, state, dir_save_file, self.AMdict["top1_target_val"].max, desc='best_prec1', smaller=False)

        self.writer.add_scalars("acc", {"val": self.AMdict["top1_target_val"].val,
                              "test": self.AMdict["top1_target_test"].val,
                              "best_prec1": self.AMdict["top1_target_test"].max}, self.epoch + 1)
        
        return self.AMdict["top1_target_test"].epochs_max_not_refresh
    
    
    
    
