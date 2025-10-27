import torch
from torch.nn import functional as F
from conformer import build_model
import numpy as np
import os
import cv2
import time
import torch.nn as nn
import argparse
import os.path as osp
import os
size_coarse = (10, 10)
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from  utils import  count_model_flops,count_model_params
from PIL import Image
import json



class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        #self.build_model()
        self.net = build_model(self.config.network, self.config.arch)
        #self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model for testing from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        if config.mode == 'train':
            if self.config.load == '':
                print("Loading pre-trained imagenet weights for fine tuning")
                self.net.BackboneExtractionModule.load_pretrained_model(self.config.pretrained_model
                                                        if isinstance(self.config.pretrained_model, str)
                                                        else self.config.pretrained_model[self.config.network])
                # load pretrained backbone
            else:
                print('Loading pretrained model to resume training')
                self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model
        
        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'GRA_Net SOD Structure')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        for p in model.parameters():
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        print(name)
        #print(model)
        print("The number of trainable parameters: {}".format(num_params_t))
        print("The number of parameters: {}".format(num_params))
        print(f'Flops: {count_model_flops(model)}')
        print(f'Flops: {count_model_params(model)}')



    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

             
                preds,_,_,_,_,_= self.net(images,depth)
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()
                #print(pred.shape)
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_convtran.png')
                cv2.imwrite(filename, multi_fuse)
                
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    

    def gradcam_pp_map(self,features, grads):

        numerator = grads.pow(2)
        denominator = 2 * grads.pow(2) + torch.sum(features * grads.pow(3), dim=(2, 3), keepdim=True)
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / denominator

        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize per image
        cam = (cam - cam.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]) / \
            (cam.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
        return cam

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        
        loss_vals=  []
        
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                    'sal_label'], data_batch['sal_edge']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label, sal_edge= sal_image.to(device), sal_depth.to(device), sal_label.to(device),sal_edge.to(device)

               
                self.optimizer.zero_grad()
                sal_label_coarse = F.interpolate(sal_label, size_coarse, mode='bilinear', align_corners=True)
                
                sal_final,coarse_sal_rgb,coarse_sal_depth,sal_edge_rgbd0,sal_edge_rgbd1,sal_edge_rgbd2,x_features, y_features = self.net(sal_image,sal_depth)
               
                x8 = x_features[8]  # CNN path
                y8 = y_features[8]  # Transformer path
                B, N, C = y8.shape
                H = W = int(N ** 0.5)
                y8 = y8[:, 1:].transpose(1, 2).unflatten(2, (20, 20))
                grad_y8 = grad_y8[:, 1:].transpose(1, 2).unflatten(2, (20, 20))
                sal_loss_coarse_rgb =  F.binary_cross_entropy_with_logits(coarse_sal_rgb, sal_label_coarse, reduction='sum')
                sal_loss_coarse_depth =  F.binary_cross_entropy_with_logits(coarse_sal_depth, sal_label_coarse, reduction='sum')
                sal_final_loss =  F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                edge_loss_rgbd0=F.smooth_l1_loss(sal_edge_rgbd0,sal_edge)
                edge_loss_rgbd1=F.smooth_l1_loss(sal_edge_rgbd1,sal_edge)
                edge_loss_rgbd2=F.smooth_l1_loss(sal_edge_rgbd2,sal_edge)
                
                sal_loss_fuse = sal_final_loss+512*edge_loss_rgbd0+1024*edge_loss_rgbd1+2048*edge_loss_rgbd2+sal_loss_coarse_rgb+sal_loss_coarse_depth
                # --- Grad-CAM++ auxiliary attention loss ---
                # 1. Backprop from saliency output to get gradients w.r.t x[8] and y[8]
                

                grad_x8 = torch.autograd.grad(sal_final.mean(), x8, retain_graph=True, create_graph=True)[0]
                grad_y8 = torch.autograd.grad(sal_final.mean(), y8, retain_graph=True, create_graph=True)[0]

                cam_x = self.gradcam_pp_map(x8, grad_x8)
                cam_y = self.gradcam_pp_map(y8, grad_y8)

                # Resize cams to saliency map size
                cam_x_up = F.interpolate(cam_x, size=sal_final.shape[2:], mode='bilinear', align_corners=False)
                cam_y_up = F.interpolate(cam_y, size=sal_final.shape[2:], mode='bilinear', align_corners=False)

                # Saliency prediction after sigmoid
                sal_sigmoid = torch.sigmoid(sal_final)

                # Compute attention alignment losses
                loss_cam_x = F.mse_loss(cam_x_up, sal_sigmoid)
                loss_cam_y = F.mse_loss(cam_y_up, sal_sigmoid)

                total_loss = sal_loss_fuse+ 0.2* (loss_cam_x + loss_cam_y)
                sal_loss = total_loss/ (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                r_sal_loss_item+=sal_loss.item() * sal_image.size(0)
                sal_loss.backward()
                self.optimizer.step()

                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %0.4f  ||sal_final:%0.4f|| edge_loss0:%0.4f|| edge_loss1:%0.4f|| edge_loss2:%0.4f|| r:%0.4f||d:%0.4f' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss,sal_final_loss,edge_loss_rgbd0,edge_loss_rgbd1,edge_loss_rgbd2,sal_loss_coarse_rgb,sal_loss_coarse_depth ))
  
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            loss_vals.append(train_loss)
            
            print('Epoch:[%2d/%2d] | Train Loss : %.3f' % (epoch, self.config.epoch,train_loss))
            import matplotlib.pyplot as plt

            if epoch % 5 == 0:
                cam_show = cam_x_up[0, 0].detach().cpu().numpy()
                plt.imshow(cam_show, cmap='jet')
                plt.title('Grad-CAM++ Attention from x[8]')
                plt.show()
        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
        

