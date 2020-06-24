import torch
import torch.nn as nn
from PIL import Image

class Grid :
    '''implementation based on https://arxiv.org/abs/2001.04086'''
    def __init__(self, dmin, dmax, rotate = 90, ratio = 0.5, mode = 1, prob = 1.):
        self.dmin = dmin
        self.dmax = dmax
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob # st prob is base prob
        
    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch/max_epoch ) # increasing probability
        
    def __call__(self,img):
        if np.random.rand() > self.prob :
            return img 
        #image must be in C X H X W format
        H,W = img.shape[1:]

        side = math.ceil((math.sqrt(H*H + W*W)))
        
        d = np.random.randint(self.dmin, self.dmax) # get random side
        
        self.l = math.ceil(d * self.ratio) # length of each droped square
        mask = np.ones((side,side), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d) # start (h,w)
        for i in range(-1, side//d + 1): # iterate over the grids
            s = d * i + st_h
            t = s + self.l 
            s = max(min(s, side), 0) # edge cases
            t = max(min(t, side) ,0)
            mask[s:t,:] = 0 # assign 0 to the non-dropped region, later negated
        
        for i in range(-1, side// d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s,side), 0)
            t = max(min(t,side), 0)
            mask[:, s:t] = 0
        
        r = np.random.randint(self.rotate)
        r = 0
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(side - H) // 2 : (side - H)//2 + H, (side - W)//2 : (side - W) // 2 + W] 
        #crop the mask that is according to the image size
        
        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask # negate mask, so the 0 covers the dropped region
            
        mask = mask.expand_as(img)
        img = img * mask
        
        return img
        
class GridMask(nn.Module): 
    def __init__(self, dmin, dmax, rotate = 90, ratio = 0.5, mode = 1, prob = 1.):
        super().__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(dmin, dmax, rotate, ratio, mode,prob)
        
    def set_prob(self, epoch, max_epoch) :
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, img): # to be applied on the fly of N X C X H X W 
        if len(img) == 4 : # meaning already passed along with data loader
            n,c,h,w = img.shape
            y = []
            for i in range(n):
                y.append(self.grid(img[i])) # apply grided pictures
                
            y = torch.cat(y).view(n,c,h,w)
            return y
        
        if len(img) == 3 : # part of torchvision transform, preferable constant prob
            c,h,w = img.shape
            return self.grid(img)
