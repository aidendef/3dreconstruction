import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict
from im2mesh.onet_IP_input import models, training, generation
from im2mesh import data
from im2mesh import config


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    encoder_latent = cfg['model']['encoder_latent']
    encoder_2 = cfg['model']['encoder_2']
    encoder_2_latent = cfg['model']['encoder_2_latent']
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']

    dim_2 = cfg['data']['dim']
    z_dim_2 = cfg['model']['z_dim_2']
    c_dim_2 = cfg['model']['c_dim_2']
    ec_dim_2 = cfg['model']['ec_dim_2']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    encoder_latent_kwargs = cfg['model']['encoder_latent_kwargs']
    encoder_2_kwargs = cfg['model']['encoder_2_kwargs']
    encoder_2_latent_kwargs = cfg['model']['encoder_2_latent_kwargs']

    decoder = models.decoder_dict[decoder](
        dim=dim, z_dim=z_dim, c_dim=c_dim,
        **decoder_kwargs
    )
    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    if z_dim != 0:
        encoder_2_latent = models.encoder_2_latent_dict[encoder_2_latent](
            dim_2=dim_2, z_dim_2=z_dim_2, c_dim_2=c_dim_2,
            **encoder_2_latent_kwargs
        )
    else:
        encoder_2_latent = None

    if encoder_2 == 'idx':
        encoder_2 = nn.Embedding(len(dataset), c_dim)
    elif encoder_2 is not None:
        encoder_2 = encoder_2_dict[encoder_2](
            c_dim=ec_dim_2,
            **encoder_2_kwargs
        )
    else:
        encoder_2 = None

    p0_z = get_prior_z(cfg, device)
    model = models.OccupancyNetwork(
        decoder, encoder, encoder_latent, encoder_2, encoder_2_latent, p0_z, device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    beta_vae = cfg['training']['beta_vae']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
        beta_vae=beta_vae,
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    preprocessor = config.get_preprocessor(cfg, device=device)

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        preprocessor=preprocessor,
    )
    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    if ((cfg['model']['encoder_latent'] == 'pointnet_conv') or 
        (cfg['model']['encoder_latent'] == 'pointnet_conv2')):

        plane_res = cfg['model']['encoder_latent_kwargs'].get('plane_resolution', 128)
        n_conv_layer = cfg['model']['encoder_latent_kwargs'].get('n_conv_layer', 4)
        plane_type = cfg['model']['encoder_latent_kwargs'].get('plane_type', ['xz'])

        latent_dim = z_dim * (2 ** n_conv_layer)
        res_dim = int(plane_res / (2 ** n_conv_layer))
        
        spatial_resolution = (res_dim, ) * 2

        p0_z = dist.Normal(
            torch.zeros((latent_dim, *spatial_resolution), device=device),
            torch.ones((latent_dim, *spatial_resolution), device=device)
        )
    else:
        p0_z = dist.Normal(
            torch.zeros(z_dim, device=device),
            torch.ones(z_dim, device=device)
        )

    return p0_z


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['model']['use_camera']
    fields = {}
    fields['points'] = data.PointsField(
        cfg['data']['points_file'], points_transform,
        with_transforms=with_transforms,
        unpackbits=cfg['data']['points_unpackbits'],
    )

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        voxels_file = cfg['data']['voxels_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
            )
        if voxels_file is not None:
            fields['voxels'] = data.VoxelsField(voxels_file)

    return fields
