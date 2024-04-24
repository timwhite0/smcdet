import torch
from torch.distributions import Poisson
from torchvision.transforms.v2.functional import gaussian_blur

class ImageAttributes(object):
    def __init__(self,
                 img_height: int,
                 img_width: int,
                 max_objects: int,
                 psf_size: int,
                 psf_stdev: float,
                 background: float):
        
        self.img_height = img_height
        self.img_width = img_width
        self.PSF_marginal_H = (torch.arange(self.img_height, dtype=torch.float32)).view(1, self.img_height, 1, 1)
        self.PSF_marginal_W = (torch.arange(self.img_width, dtype=torch.float32)).view(1, 1, self.img_width, 1)
        
        self.max_objects = max_objects
        
        self.psf_size = psf_size
        self.psf_stdev = psf_stdev
        self.background = background
    
    
    def Ellipse(self, num_layers, locs, axes, angles):
        loc_H = locs[:,:,0].view(-1, 1, 1, num_layers)
        loc_W = locs[:,:,1].view(-1, 1, 1, num_layers)
        axis_H = axes[:,:,0].view(-1, 1, 1, num_layers)
        axis_W = axes[:,:,1].view(-1, 1, 1, num_layers)
        cos_angles = torch.cos(angles).view(-1, 1, 1, num_layers)
        sin_angles = torch.sin(angles).view(-1, 1, 1, num_layers)
        
        x = torch.arange(0., self.img_height)
        y = torch.arange(0., self.img_height)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X = X.view(1, self.img_height, self.img_width, 1)
        Y = Y.view(1, self.img_height, self.img_width, 1)
        
        Xrot = ((X - loc_H) * cos_angles) - ((Y - loc_W) * sin_angles) + loc_H
        Yrot = ((X - loc_H) * sin_angles) + ((Y - loc_W) * cos_angles) + loc_W

        ellipse_mask = 1*(((Xrot - loc_H) / axis_H) ** 2 + ((Yrot - loc_W) / axis_W) ** 2 <= 1)
        
        return ellipse_mask
    
    
    def tileEllipse(self, tile_slen, num_layers, locs, axes, angles):
        loc_H = locs[:,:,:,:,0].view(locs.shape[0], locs.shape[1], -1, 1, 1, num_layers)
        loc_W = locs[:,:,:,:,1].view(locs.shape[0], locs.shape[1], -1, 1, 1, num_layers)
        axis_H = axes[:,:,:,:,0].view(axes.shape[0], axes.shape[1], -1, 1, 1, num_layers)
        axis_W = axes[:,:,:,:,1].view(axes.shape[0], axes.shape[1], -1, 1, 1, num_layers)
        cos_angles = torch.cos(angles).view(angles.shape[0], angles.shape[1], -1, 1, 1, num_layers)
        sin_angles = torch.sin(angles).view(angles.shape[0], angles.shape[1], -1, 1, 1, num_layers)
        
        x = torch.arange(0., tile_slen)
        y = torch.arange(0., tile_slen)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X = X.view(1, 1, 1, tile_slen, tile_slen, 1)
        Y = Y.view(1, 1, 1, tile_slen, tile_slen, 1)
        
        Xrot = ((X - loc_H) * cos_angles) - ((Y - loc_W) * sin_angles) + loc_H
        Yrot = ((X - loc_H) * sin_angles) + ((Y - loc_W) * cos_angles) + loc_W

        ellipse_mask = 1*(((Xrot - loc_H) / axis_H) ** 2 + ((Yrot - loc_W) / axis_W) ** 2 <= 1)
        
        return ellipse_mask


    def PSF(self, num_layers, locs):
        loc_H = locs[:,:,0].view(-1, 1, 1, num_layers)
        loc_W = locs[:,:,1].view(-1, 1, 1, num_layers)
        logpsf = (-(self.PSF_marginal_H - loc_H)**2 - (self.PSF_marginal_W - loc_W)**2)/(2*(self.psf_stdev**2))
        psf = (logpsf).exp()
        
        return psf
    
    
    def tilePSF(self, num_layers, locs):
        loc_H = locs[:,:,:,:,0].view(locs.shape[0], locs.shape[1], -1, 1, 1, num_layers)
        loc_W = locs[:,:,:,:,1].view(locs.shape[0], locs.shape[1], -1, 1, 1, num_layers)
        logpsf = (-(self.PSF_marginal_H - loc_H)**2 - (self.PSF_marginal_W - loc_W)**2)/(2*self.psf_stdev**2)
        psf = (logpsf).exp()
        
        return psf
    
    
    def generate(self,
                 prior,
                 num = 1):
        
        counts, fluors, locs, axes, angles = prior.sample(num_catalogs = num)
        
        cell_intensities = fluors.view(num, 1, 1, self.max_objects) * self.Ellipse(self.max_objects, locs, axes, angles)
        cell_intensities = (cell_intensities).sum(3)
        
        total_intensities = gaussian_blur(cell_intensities + self.background,
                                          kernel_size = self.psf_size, sigma = self.psf_stdev)
        total_intensities *= torch.normal(torch.ones_like(cell_intensities),
                                          0.05*torch.ones_like(cell_intensities)).clamp(min=1e-1)
        
        images = Poisson(total_intensities).sample()
        
        return [counts, fluors, locs, axes, angles, total_intensities, images]