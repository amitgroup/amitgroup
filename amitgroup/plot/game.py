
import pygame
import pygame.locals as pylocals
import numpy as np

class PlottingWindow(object):
    def __init__(self, figsize=(8, 6), subplots=(1, 1), caption="Plotting Window"):
        pygame.init() 
        #if not pygame.font: print 'Warning, fonts disabled'
        self._window_size = tuple(map(lambda x: x*80, figsize))
        self._screen = pygame.display.set_mode(self._window_size)
        pygame.display.set_caption(caption)
        pygame.mouse.set_visible(1)

        self._clock = pygame.time.Clock()

        self._background = pygame.Surface(self._screen.get_size())
        self._background = self._background.convert()
        self._background.fill((130, 130, 130))

        i = 0
        self._quit = False
        self.subplots(subplots)

    def subplots(self, shape):
        assert len(shape) == 2, "Subplots shape must have two elemements"
        self._subplots = shape

    def tick(self, clear=True):
        if self._quit:
            return False
        #clock.time(60)  
        for event in pygame.event.get():
            if event.type == pylocals.QUIT or \
               event.type == pylocals.KEYDOWN and event.key == pylocals.K_ESCAPE:
                self._quit = True
                pygame.display.quit()
                return False 

        if clear:
            self._screen.blit(self._background, (0, 0))

        #if pygame.font:
        #    font = pygame.font.Font(None, 36)
        #    text = font.render("x", 1, (10, 10, 10))
        #    textpos = text.get_rect(centerx=_background.get_width()/2)
        #    _screen.blit(text, textpos)

        #pygame.display.flip()
        return True 

    def _anchor_and_size(self, subplot):
        pad = 10 
        p = (subplot%self._subplots[1], subplot//self._subplots[1])
        size = (self._window_size[0]//self._subplots[1], self._window_size[1]//self._subplots[0])
        anchor = (p[0] * size[0], p[1] * size[1])

        return (anchor[0]+pad, anchor[1]+pad), (size[0]-2*pad, size[1]-2*pad) 

    def imshow(self, im, limits=(0, 1), subplot=0, caption=None):
        assert isinstance(im, np.ndarray) and len(im.shape) == 2, "Image must be a 2D ndarray"
        anchor, size = self._anchor_and_size(subplot)

        # Normalize
        if limits != (0, 1):
            span = (limits[1]-limits[0])
            if span == 0.0:
                return
            im2 = (im-limits[0])/span
        else:
            im2 = im

        im3 = (np.clip(im2.T, 0, 1)*255).astype(np.uint32)
        scr = pygame.Surface(im.shape)
        pygame.surfarray.blit_array(scr, (im3<<24) + (im3<<16) + (im3<<8) + 0xFF)
        scale = min((size[0])//im.shape[0], (size[1])//im.shape[1])
        scr2 = pygame.transform.scale(scr, (im.shape[0]*scale, im.shape[1]*scale))
        #_screen.blit(scr2, (640//2 - im.shape[0]*scale//2, 0))
        self._screen.blit(scr2, anchor)

        if caption and pygame.font:
            font = pygame.font.Font(None, 16)
            text = font.render(caption, 1, (10, 10, 10))
            self._screen.blit(text, (anchor[0], anchor[1]-10))

    def plot(self, x, y=None, limits='auto', subplot=0):
        N = len(x) 
        if N < 2:
            return # Just don't draw anything
        anchor, size = self._anchor_and_size(subplot)
        if y is None:
            y = x
            x = np.arange(N)
        x = np.asarray(x)
        y = np.asarray(y)
        if limits == 'auto':
            xlims = x.min(), x.max()
            ylims = y.min(), y.max()
    
        @np.vectorize
        def x2pixel(x0):
            return anchor[0] + size[0]*(x0-xlims[0])/(xlims[1]-xlims[0])
        @np.vectorize
        def y2pixel(y0):
            return anchor[1] + size[1]*(1-((y0-ylims[0])/(ylims[1]-ylims[0])))

        px = x2pixel(x)
        py = y2pixel(y)
        pointlist = zip(px, py)
        pygame.draw.aalines(self._screen, (255, 255, 255), False, pointlist)

        if pygame.font: 
            font = pygame.font.Font(None, 16)
            text = font.render("{0:.1g}/{1:.1g}".format(*ylims), 1, (10, 10, 10))
            self._screen.blit(text, (anchor[0], anchor[1]-10))

    def flip(self, fps=0):
        pygame.display.flip()
        if fps > 0:
            self._clock.tick(fps)

    def mainloop(self, fps=60):
        while self.tick(clear=False):
            self.flip(fps)
