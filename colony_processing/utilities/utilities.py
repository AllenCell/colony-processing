import numpy as np


def create_circular_mask(img_shape, center=None, radius=None):
    """
    Creates a circular mask over an image, masking out edges of a well
    Parameters
    ----------
    img_shape           a tuple of (image height, image width)
    center              set center location of mask
    radius              radius of mask

    Returns
    -------
    mask                a binary mask
    """
    height, width = img_shape
    if center is None:  # use the middle of the image
        center = [int(width / 2), int(height / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], width - center[0], height - center[1])

    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center >= radius
    return mask
