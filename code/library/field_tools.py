# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:32:33 2020

@author: premv
"""

import numpy as np
from matplotlib import widgets, pyplot as plt
import pandas as pd

def compute_power_spec(FX):
    FK = np.fft.rfftn(FX)
    Pk = pd.DataFrame(data=np.vstack((self.k.ravel(),
                    np.abs(self.FK_computed.ravel())**2)).T,columns=['k','Pk']).sort_values('k')

def power_law(b,n):
        """Returns a function which is a power law in one variable."""
        return lambda k : b*k**n

class Field():
    def __init__(self,FX,box_size=150):
        self.FX = FX
        self.shape = FX.shape
        self.box_size = box_size
        self.cell_size = self.box_size/self.shape[0]
        
    
    def generate_from_Pk(self,shape):
        self.shape = shape
        self.K = np.array(np.meshgrid(*[np.fft.fftfreq(x) * x for x in self.shape[:-1]],
                                         np.fft.rfftfreq(self.shape[-1]) * self.shape[-1]))
#        self.K = self.K1.transpose(0,*np.arange(len(self.shape),0,-1))
        self.k = np.sqrt((self.K**2).sum(axis=0))    # magnitude of K vector
        self.Pk = self.P(self.k)    
#        np.random.seed(840900)
        self.FK = (np.random.randn(*self.k.shape) + np.random.randn(*self.k.shape) *1j) * np.sqrt(self.Pk/2)
        self.FX = np.fft.irfftn(self.FK,shape)
        
    def set_field(self,grid):
        self.FX = grid
        self.shape = grid.shape
        
    def visualise(self,title="The gaussian random field in physical 3D space"):
#        plt.figure(dpi=120)
#       plt.imshow(FX[1])
        self.visual_fig,self.visual_axis = plt.subplots(dpi=120)
        self.interact = self.visual_axis.imshow(self.FX[:,:,0]) #shows 0th frame
        self.visual_axis.set_title(title)
        self.visual_fig.subplots_adjust(left=0.25, bottom=0.25)
        axframe = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.visual_slider = widgets.Slider(axframe, 'third direction', 0, self.shape[-1]-1, valinit=0)
        
        def update(val):
#            print(val)
            self.interact.set_data(self.FX[:,:,int(val)])
        
        self.visual_slider.on_changed(update)
    
    def compute_Pk_from_field(self):
        self.FK_computed = np.fft.rfftn(self.FX)
        self.Pk_computed = pd.DataFrame(data=np.vstack((self.k.ravel(),
                    np.abs(self.FK_computed.ravel())**2)).T,columns=['k','Pk']).sort_values('k')
        
    def plot_original_and_computed_Pk(self,bins):
        self.plot_fig,self.plot_axis = plt.subplots(dpi=120)
        self.grouped3 = self.Pk_computed.groupby(pd.cut(self.Pk_computed['k'], bins=bins)).mean()
        self.plot_axis.plot(self.grouped3['k'],self.P(self.grouped3['k']),'-',label='Used to generate the random field')
        self.plot_axis.plot(self.grouped3['k'],self.grouped3['Pk'],'-',label='Computed from the generated random field')
        self.plot_axis.set_xscale('log')
        self.plot_axis.set_yscale('log')
        self.plot_axis.set_ylim(bottom=1)
        self.plot_axis.set_title("Power spectrum P(k)")
        self.plot_axis.legend(loc="upper left")
        
        
        
        
        
        
        
        
        
        
        