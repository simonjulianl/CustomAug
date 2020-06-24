import numpy as np 
import torch 

class Cutout(torch.nn.Module):
    """pytorch implementation of Cutoff based on https://arxiv.org/abs/1708.04552
    """
    def __init__(self, length, n_holes = 1):
        super().__init__()
        self.n_holes = n_holes # by default its 1
        self.length = length # patches of length x length

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W). for RGB or (H,W) for BLACK_WHITE
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        
        return self.patch_image(img)
    
    def __repr__(self):
        return self.__class__.__name__ + '(n_holes={}, length = {})'.format(self.n_holes, self.length)
    
    def patch_image(self,img):
        if len(img.shape) == 3 : #RGB for C X H X W ( torch format )
            H,W = img.shape[1:]
        else : # BLACK_WHITE
            H,W = img.shape
            
        mask = torch.ones((H,W)) 
        
        for i in range(self.n_holes):
            y = np.random.randint(H) # get centre (x,y)
            x = np.random.randint(W)
            
            y1 = np.clip(y - self.length // 2, 0, H) # clip beyond edges
            y2 = np.clip(y + self.length // 2, 0, H)
            x1 = np.clip(x - self.length // 2, 0, W) # clip beyond edges
            x2 = np.clip(x + self.length // 2, 0, W)
            
            mask[y1:y2, x1: x2] = 0
        
        mask = mask.expand_as(img)
        img = img * mask # mask it with to become zero
        
        return img 
