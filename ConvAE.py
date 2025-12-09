import torch
import torch.nn as nn
import pdb

class Conv2DAutoEncoder(nn.Module):
	def __init__(self, input_shape):
		super(Conv2DAutoEncoder, self).__init__()
		
		self.encoder = nn.Sequential(
			# Conv1 -> BN -> RELU  -> MaxPool
			nn.Conv2d(input_shape, 512, 11, stride=4),   # 204->49
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2,stride=2),    # 49->24
			
            # Conv2 -> BN -> RELU -> Max Pool
			nn.Conv2d(512,256,5,stride =1,padding=2),  # 24->24
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2,stride =2),                  # 24->12
			
            # Conv3 -> BN -> RELU
            nn.Conv2d(256,128,3,stride =1,padding=1),  # 12->12
		    nn.BatchNorm2d(128),
			nn.ReLU(inplace=True)
			
        )
		self.decoder = nn.Sequential(
			# Deconv Layer1
			nn.ConvTranspose2d(128,128,3,stride=1,padding=1),   # 12->12
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(128,128,2,stride=2,dilation=2),  # 12->24
			
            nn.ConvTranspose2d(128,256,3,stride=1,padding=1),   # 24->24
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(256,256,2,stride=2, dilation=2),  # 24->48
			
            nn.ConvTranspose2d(256,512,5,stride=1, padding=2),   # 48->48
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(512, input_shape, 11, stride=4, padding=2, output_padding=1)   # 48->204

        )
	def forward(self,img):
		img = self.encoder(img)
		img = self.decoder(img)
		return img.contiguous()
