import torch
import math
import torch.nn.functional as F


def oriented_sine_grating(size, frequency, orientation, contrast, phase=0):
    """
    Create an oriented sine wave grating
    
    Args:
        size: image size (height, width) or single int for square
        frequency: spatial frequency (cycles per pixel)
        orientation: orientation in radians (0 = horizontal)
        phase: phase offset in radians
    """
    if isinstance(size, int):
        h, w = size, size
    else:
        h, w = size
    
    # Create coordinate grids centered at middle
    y = torch.arange(h, dtype=torch.float32) - h // 2
    x = torch.arange(w, dtype=torch.float32) - w // 2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Rotate coordinates according to orientation
    cos_ori = math.cos(orientation)
    sin_ori = math.sin(orientation)
    X_rot = X * cos_ori + Y * sin_ori
    
    # Create sine wave grating
    grating = torch.sin(2 * math.pi * frequency * X_rot + phase)

    grating *= contrast
    
    return grating



def _get_distance_from_center(size, center=None):
    if isinstance(size, int):
        h, w = size, size
    else:
        h, w = size
    
    if center is None:
        center = (h // 2, w // 2)
    
    y = torch.arange(h, dtype=torch.float32) - center[0]
    x = torch.arange(w, dtype=torch.float32) - center[1]
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    distance = torch.sqrt(X**2 + Y**2)
    return distance


def circular_aperture_mask(size, radius, center=None):
    """
    Create a circular aperture mask
    
    Args:
        size: image size
        radius: radius of circular aperture
        center: center of circle, defaults to image center
    """
    distance = _get_distance_from_center(size, center=center)
    mask = (distance <= radius).float()
    
    return mask



def gaussian_aperture(size, radius, center=None):
    """
    To make Gabor from the gratings image.
    """

    distance = _get_distance_from_center(size, center=center)
    sigma = radius / 2
    mask = torch.exp(-0.5 * (distance / sigma)**2)

    return mask




def apply_random_translation(frames, translation_std=2.0):
    """
    Apply random translation to a batch of frames
    
    Args:
        frames: tensor of shape (n_frames, channels, height, width)
        max_shift: maximum translation in pixels
    
    Returns:
        translated frames
    """

    n_frames, channels, height, width = frames.shape
    
    # Generate random translations for each frame
    translations = torch.randn(n_frames, 2) * translation_std  # (n_frames, 2)
    
    # Normalize to [-1, 1] range (grid_sample expects this)
    translations[:, 0] /= (width / 2)   # x translation
    translations[:, 1] /= (height / 2)  # y translation
    
    # Create affine transformation matrices (just translation)
    theta = torch.zeros(n_frames, 2, 3)
    theta[:, 0, 0] = 1  # scale x = 1
    theta[:, 1, 1] = 1  # scale y = 1
    theta[:, 0, 2] = translations[:, 0]  # translate x
    theta[:, 1, 2] = translations[:, 1]  # translate y
    
    # Generate sampling grid
    grid = F.affine_grid(theta, frames.size(), align_corners=False)
    
    # Apply transformation
    translated_frames = F.grid_sample(frames, grid, mode='bilinear', 
                                    padding_mode='border', align_corners=False)
    
    return translated_frames

