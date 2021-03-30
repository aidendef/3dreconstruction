import os
from tqdm import trange
import torch
from matplotlib import pyplot as plt

import numpy as np
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
import im2mesh.common as common
# from im2mesh.common import (
#     Projection2d)


class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        
        # self.p2d = Projection2d()
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        world_mat = data.get('inputs.world_mat').to(device)
        camera_mat = data.get('inputs.camera_mat').to(device)
        camera_args = common.get_camera_args(
            data, 'points.loc', 'points.scale', device=self.device)
        world_mat, camera_mat = camera_args['Rt'], camera_args['K']

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs,world_mat, camera_mat, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs, world_mat, camera_mat,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs, world_mat, camera_mat,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        world_mat = data.get('inputs.world_mat').to(device)
        camera_mat = data.get('inputs.camera_mat').to(device)
        camera_args = common.get_camera_args(
            data, 'points.loc', 'points.scale', device=self.device)
        world_mat, camera_mat = camera_args['Rt'], camera_args['K']

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            p_r = self.model(p, inputs, world_mat, camera_mat, sample=self.eval_sample, **kwargs)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def colornoise(self, img, npp):
        device = self.device
        bat = img.size(0)
        noise = torch.randn(bat, 3, 224, 224).to(device)
        img = img + (noise*npp)
        return img

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''

        npp= 0.1

        device = self.device
        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        inputs = self.colornoise(inputs, npp)


        world_mat = data.get('inputs.world_mat').to(device)
        camera_mat = data.get('inputs.camera_mat').to(device)
        camera_args = common.get_camera_args(
            data, 'points.loc', 'points.scale', device=self.device)
        world_mat, camera_mat = camera_args['Rt'], camera_args['K']
        
        self.vis(False, inputs[0].cpu().numpy())
        # print("world_mat",world_mat.shape)
        # print("camera_mat",camera_mat.shape)
        # exit(1)

        kwargs = {}

        G, c = self.model.encode_inputs(inputs)
        # print("c0",c[0].shape) 64, 56, 56
        # print("c1",c[1].shape) 128, 28, 28
        # print("c2",c[2].shape) 256, 14, 14
        # print("c3",c[3].shape) 512, 7, 7
        # print("c4",c[4].shape) 256, 2, 2
        # print("G",G.shape) 1024
        v = self.model.gproj(p, G, c, world_mat, camera_mat, inputs, False)
        # v = self.model.gproj(p, c, camera_mat, inputs, True)
        # v point number, 1219       
        # v+G point number, 2243
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        logits = self.model.decode(p, z, v, **kwargs).logits
        
        # exit(1)
        
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        return loss

    def vis(self, flag, img, output_file='/home/hpclab/kyg/occupancy_networks/test/gp_test.png'):
        if flag == True :
            plt.imshow(img.transpose(1, 2, 0))
            # plt.plot(
            #     (points_img[:, 0] + 1)*img.shape[1]/2,
            #     (points_img[:, 1] + 1) * img.shape[2]/2, 'x')
            
            # plt.savefig(output_file)
            plt.savefig(output_file)
