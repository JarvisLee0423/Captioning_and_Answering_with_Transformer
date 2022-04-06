'''
    Copyright:      JarvisLee
    Date:           2022/01/17
    Filename:       Layers.py
    Description:    Implement the necessary layers of the Captioning and Answering with Transformer.
'''

# Import the necessary library.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implement the outlook-attention.
class OutlookAttention(nn.Module):
    '''
        Implement the outlook-attention.\n
        Params:\n
            - 'hiddenSize' is the dimension of the input hidden data.
            - 'headSize' is the number of the attention heads.
            - 'kernelSize' is the size of attention window of the outlook attention.
            - 'padding' is the padding of attention window of the outlook attention.
            - 'stride' is the stride of attention window of the outlook attention.
            - 'attenDrop' is the drop rate of the attention dropout.
            - 'projDrop' is the drop rate of the linear dropout.
    '''
    
    # Create the constructor.
    def __init__(self, hiddenSize, headSize, kernelSize = 3, padding = 1, stride = 1, attenDrop = 0.0, projDrop = 0.0):
        # Create the super constructor.
        super(OutlookAttention, self).__init__()
        # Split the whole hidden size into each head.
        headHiddenSize = hiddenSize // headSize
        # Get the attention normalizing scale.
        self.scale = headHiddenSize ** -0.5
        # Get the memeber variables.
        self.headSize = headSize
        self.kernelSize = kernelSize
        self.padding = padding
        self.stride = stride
        # Set the linear layer to get the value of the attention.
        self.valueLayer = nn.Linear(hiddenSize, hiddenSize, bias = False)
        # Set the linear layer to directly get the attention score in the outlook attention.
        self.attenScore = nn.Linear(hiddenSize, kernelSize ** 4 * headSize)
        # Set the attention dropout.
        self.attenDrop = nn.Dropout(attenDrop)
        # Set the linear layer to project the output.
        self.projLayer = nn.Linear(hiddenSize, hiddenSize)
        # Set the projection dropout.
        self.projDrop = nn.Dropout(projDrop)
        # Set the layer to unfold the input data and generate the value of the outlook attention.
        self.unfold = nn.Unfold(kernel_size = kernelSize, padding = padding, stride = stride)
        # Set the average pooling to aggregate the output of the outlook attention.
        self.avgPool = nn.AvgPool2d(kernel_size = stride, stride = stride, ceil_mode = True)
    
    # Create the forward.
    def forward(self, x):
        # Get the shape of the input data.
        B, C, H, W = x.shape
        # Get the height and width according to the stride.
        height, width = math.ceil(H / self.stride), math.ceil(W / self.stride)
        # Prepare the value of the outlook attention.
        value = self.valueLayer(x.permute(0, 2, 3, 1)).permute(0, 3, 2, 1)
        # Unfold and generate the value of the outlook attention. [B, H, W, C] -> [B, C*K*K, H*W] -> [B, Head, H*W, K*K, C/Head]
        value = self.unfold(value).reshape(B, self.headSize, C // self.headSize, self.kernelSize * self.kernelSize, height * width).permute(0, 1, 4, 3, 2)
        # Prepare the attention. [B, C, H, W] -> [B, H, W, C] -> [B, H, W, K*K*K*K*Head] -> [B, Head, H*W, K*K, K*K]
        atten = self.avgPool(x).permute(0, 2, 3, 1)
        atten = self.attenScore(atten).reshape(B, height * width, self.headSize, self.kernelSize * self.kernelSize, self.kernelSize * self.kernelSize).permute(0, 2, 1, 3, 4)
        atten = atten * self.scale
        # Compute the attention.
        atten = F.softmax(atten, dim = -1)
        atten = self.attenDrop(atten)
        # Compute the outlook attention. [B, Head, H*W, K*K, C/Head] -> [B, Head, C/Head, K*K, H*W] -> [B, C*K*K, H*W]
        x = (atten @ value).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernelSize * self.kernelSize, height * width)
        # Fold the attention result into the output. [B, C*K*K, H*W] -> [B, C, H, W]
        x = F.fold(x, output_size = (H, W), kernel_size = self.kernelSize, padding = self.padding, stride = self.stride)
        # Compute the output. [B, C, H, W] -> [B, H, W, C]
        output = self.projLayer(x.permute(0, 2, 3, 1))
        output = self.projDrop(output)
        # Return the output. [B, H, W, C] -> [B, C, H, W]
        return output.permute(0, 3, 1, 2)

# Implement the self-attention.
class SelfAttention(nn.Module):
    '''
        Implement the self-attention.\n
        Params:\n
            - 'hiddenSize' is the dimension of the hidden state.
            - 'bertHiddenSize' is the dimension of the hidden state of the Bidirection Encoder Representation from Transformer.
            - 'headSize' is the number of the attention heads.
            - 'attenDrop' is the drop rate of the attention dropout.
            - 'projDrop' is the drop rate of the linear dropout.
            - 'maskAtten' is the boolean to indicate whether apply the masked-self-attention.
            - 'crossAtten' is the boolean to indicate whether apply the cross-self-attention.
            - 'word2vec' is the boolean to indicate whether apply the traditional word embedding.
    '''

    # Create the constructor.
    def __init__(self, hiddenSize, bertHiddenSize, headSize, attenDrop = 0.0, projDrop = 0.0, maskAtten = False, crossAtten = False, word2vec = False):
        # Create the super constructor.
        super(SelfAttention, self).__init__()
        # Split the whole hidden size into each head.
        headHiddenSize = hiddenSize // headSize
        # Get the attention normalizing scale.
        self.scale = headHiddenSize ** -0.5
        # Get the head size.
        self.headSize = headSize
        # Get the hidden size.
        self.hiddenSize = hiddenSize
        # Check whether the attention is the masked-self-attention.
        if maskAtten:
            # Set the state of the self-attention.
            self.state = 'maskAtten'
            # Check whether applied the traditional word embedding.
            if word2vec:
                # Set the linear layer to get the query, key and value.
                self.qkv = nn.Linear(hiddenSize, hiddenSize * 3, bias = False)
                # Set the linear layer to project the output.
                self.projLayer = nn.Linear(hiddenSize, hiddenSize)
            else:
                # Set the linear layer to get the query, key and value.
                self.qkv = nn.Linear(bertHiddenSize, hiddenSize * 3, bias = False)
                # Set the linear layer to project the output.
                self.projLayer = nn.Linear(hiddenSize, bertHiddenSize)
        # Check whether the attention is the cross-self-attention.
        elif crossAtten:
            # Set the state of the self-attention.
            self.state = 'crossAtten'
            # Check whether applied the traditional word embedding.
            if word2vec:
                # Set the query.
                self.q = nn.Linear(hiddenSize, hiddenSize, bias = False)
                # Set the linear layer to project the output.
                self.projLayer = nn.Linear(hiddenSize, hiddenSize)
            else:
                # Set the query.
                self.q = nn.Linear(bertHiddenSize, hiddenSize, bias = False)
                # Set the linear layer to project the output.
                self.projLayer = nn.Linear(hiddenSize, bertHiddenSize)
            # Set the key and value.
            self.kv = nn.Linear(hiddenSize, hiddenSize * 2, bias = False)
        # Check whether the attention is the self-attention.
        else:
            # Set the state of the self-attention.
            self.state = 'selfAtten'
            # Set the linear layer to get the query, key and value.
            self.qkv = nn.Linear(hiddenSize, hiddenSize * 3, bias = False)
            # Set the linear layer to project the output.
            self.projLayer = nn.Linear(hiddenSize, hiddenSize)
        # Set the attention dropout.
        self.attenDrop = nn.Dropout(attenDrop)
        # Set the projection dropout.
        self.projDrop = nn.Dropout(projDrop)
    
    # Generate the attention mask.
    def getAttenMask(data):
        '''
            Get the attention mask.\n
            Params:\n
                - 'data' is the input data which is used to generate the mask.
        '''
        # Get the sequence length.
        seqLen = data.shape[1]
        # Create the mask.
        mask = (1 - torch.triu(torch.ones((1, 1, seqLen, seqLen), device = data.device), diagonal = 1)).bool()
        # Return the mask. (mask.shape = [B, head, seqLen, seqLen])
        return mask.to(data.device)
    
    # Create the forward.
    def forward(self, x, padMask = None, attenMask = None):
        # Check the self-attention state.
        if self.state == 'maskAtten':
            # Get the shape of the input data.
            B, L, _ = x.shape
            C = self.hiddenSize
            # Get the query, key and value. [B, L, C] -> [B, L, C*3] -> [B, L, 3, Head, C/Head] -> [3, B, Head, L, C/Head]
            qkv = self.qkv(x).reshape(B, L, 3, self.headSize, C // self.headSize).permute(2, 0, 3, 1, 4)
            # Get the query, key and value. [3, B, Head, L, C/Head] -> ([B, Head, L, C/Head], [B, Head, L, C/Head], [B, Head, L, C/Head])
            q, k, v = qkv[0], qkv[1], qkv[2]
            # Compute the attention. [B, Head, L, C/Head] -> [B, Head, L, L]
            atten = (q @ k.transpose(2, 3)) * self.scale
            # Check whether input the masks.
            assert attenMask is not None and padMask is not None, 'Please input the attention and padding masks before apply the masked-self-attention!'
            # Combine the padding mask and attention mask.
            mask = (padMask.unsqueeze(-2) & attenMask.squeeze(1)).unsqueeze(1)
            # Mask the future tokens.
            atten = atten.masked_fill(mask == 0, -1e9)
            atten = F.softmax(atten, dim = -1)
            atten = atten.masked_fill(padMask.unsqueeze(-1).unsqueeze(1) == 0, 0)
            atten = self.attenDrop(atten)
            # Compute the output. [B, Head, L, L] -> [B, Head, L, C/Head] -> [B, L, Head, C/Head] -> [B, L, C]
            output = (atten @ v).transpose(1, 2).reshape(B, L, C)
            output = self.projLayer(output)
            output = self.projDrop(output)
            # Return the output.
            return output
        elif self.state == 'crossAtten':
            # Check whether input the cross data.
            assert len(x) == 2, 'Please input a tuple which contains the vision and linguistic features!'
            # Get the vision input and the linguistic input.
            vision, linguistic = x
            # Get the shape of the vision input.
            B, Lv, C = vision.shape
            # Get the shape of the linguistic input.
            _, Ll, _ = linguistic.shape
            # Compute the query. [B, Ll, C] -> [B, Ll, C] -> [B, Head, Ll, C/Head]
            q = self.q(linguistic).reshape(B, self.headSize, Ll, C // self.headSize)
            # Compute the key and value. [B, Lv, C] -> [B, Lv, C*2] -> [B, Lv, 2, Head, C/Head] -> [2, B, Head, Lv, C/Head]
            kv = self.kv(vision).reshape(B, Lv, 2, self.headSize, C // self.headSize).permute(2, 0, 3, 1, 4)
            # Get the key and value. [2, B, Head, Lv, C/Head] -> ([B, Head, Lv, C/Head], [B, Head, Lv, C/Head])
            k, v = kv[0], kv[1]
            # Compute the attention. [B, Head, Ll, C/Head] * [B, Head, C/Head, Lv] -> [B, Head, Ll, Lv]
            atten = (q @ k.transpose(2, 3)) * self.scale
            # Check whether input the masks.
            assert padMask is not None, 'Please input the padding masks before apply the cross-self-attention!'
            # Mask the pad tokens.
            atten = F.softmax(atten, dim = -1)
            atten = atten.masked_fill(padMask.unsqueeze(-1).unsqueeze(1) == 0, 0)
            atten = self.attenDrop(atten)
            # Compute the output. [B, Head, Ll, Lv] * [B, Head, Lv, C/Head] -> [B, Head, Ll, C/Head] -> [B, Ll, Head, C/Head] -> [B, Ll, C]
            output = (atten @ v).transpose(1, 2).reshape(B, Ll, C)
            output = self.projLayer(output)
            output = self.projDrop(output)
            # Return the output.
            return output
        else:
            # Get the shape of the input data.
            B, C, H, W = x.shape
            # Compute the query, key and value. [B, C, H, W] -> [B, H, W, C] -> [B, H, W, C*3] -> [B, H*W, 3, Head, C/Head] -> [3, B, Head, H*W, C/Head]
            qkv = self.qkv(x.permute(0, 2, 3, 1)).reshape(B, H * W, 3, self.headSize, C // self.headSize).permute(2, 0, 3, 1, 4)
            # Get the query, key and value. [3, B, Head, H*W, C/Head] -> ([B, Head, H*W, C/Head], [B, Head, H*W, C/Head], [B, Head, H*W, C/Head])
            q, k, v = qkv[0], qkv[1], qkv[2]
            # Compute the attention. [B, Head, H*W, C/Head] -> [B, Head, H*W, H*W]
            atten = (q @ k.transpose(2, 3)) * self.scale
            atten = F.softmax(atten, dim = -1)
            atten = self.attenDrop(atten)
            # Compute the output. [B, Head, H*W, H*W] -> [B, Head, H*W, C/Head] -> [B, H*W, Head, C/Head] -> [B, H, W, C]
            output = (atten @ v).transpose(1, 2).reshape(B, H, W, C)
            output = self.projLayer(output)
            output = self.projDrop(output)
            # Return the output. [B, H, W, C] -> [B, C, H, W]
            return output.permute(0, 3, 1, 2)

# Implement the multi-layer perceptron.
class MLP(nn.Module):
    '''
        Implement the multi-layer perceptron.\n
        Params:\n
            - 'inputSize' is the size of the input data.
            - 'hiddenSize' is the size of the hidden data.
            - 'projDrop' is the drop rate of the linear dropout.
            - 'decoder' is the boolean to indicate whether the transformer block is part of the decoder.
    '''

    # Create the constructor.
    def __init__(self, inputSize, hiddenSize, projDrop = 0.0, decoder = False):
        # Create the super constructor.
        super(MLP, self).__init__()
        # Get the decoder flag.
        self.decoder = decoder
        # Set the first linear layer to project the output.
        self.projLayer1 = nn.Linear(inputSize, hiddenSize)
        # Set the activation function.
        self.gelu = nn.GELU()
        # Set the second linear layer to project the output.
        self.projLayer2 = nn.Linear(hiddenSize, inputSize)
        # Set the projection dropout.
        self.projDrop = nn.Dropout(projDrop)
    
    # Create the forward.
    def forward(self, x):
        # Check the decoder flag.
        if self.decoder:
            # Compute the output. [B, L, C]
            x = self.projLayer1(x)
            x = self.gelu(x)
            x = self.projDrop(x)
            x = self.projLayer2(x)
            output = self.projDrop(x)
            # Return the output. [B, L, C]
            return output
        else:
            # Compute the output. [B, C, H, W] -> [B, H, W, C]
            x = self.projLayer1(x.permute(0, 2, 3, 1))
            x = self.gelu(x)
            x = self.projDrop(x)
            x = self.projLayer2(x)
            output = self.projDrop(x)
            # Return the output. [B, H, W, C] -> [B, C, H, W]
            return output.permute(0, 3, 1, 2)

# Implement the positional embedding.
class PosEmbed(nn.Module):
    '''
        Implement the cos-sin positional embedding.\n
    '''
    
    # Create the constructor.
    def __init__(self):
        # Create the super constructor.
        super(PosEmbed, self).__init__()
    
    # Create the forward.
    def forward(self, x):
        # Initialize the positional embedding.
        PE = []
        # Get the positional embedding.
        for pos in range(x.shape[1]):
            PE.append([pos / np.power(10000, (2 * (i // 2) / x.shape[2])) for i in range(x.shape[2])])
        # Convert the positional embedding to be the tensor.
        PE = torch.tensor(PE, dtype = torch.float32)
        # Compute the positional embedding.
        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        # Return the positional embedding. [1, seqLen, dim]
        return PE.unsqueeze(0).to(x.device).detach()

# Implement the patch embedding.
class PatchEmbed(nn.Module):
    '''
        Implement the patch embedding.\n
        Params:\n
            - 'inChannel' is the number of the channels of the input data.
            - 'hiddenSize' is the size of the hidden data.
            - 'embedSize' is the size of the patch embedding.
            - 'patchSize' is the size of each patch.
            - 'stride' is the size of the stride.
    '''

    # Create the constructor.
    def __init__(self, inChannel = 3, hiddenSize = 64, embedSize = 384, patchSize = 8, stride = 1):
        # Create the super constructor.
        super(PatchEmbed, self).__init__()
        # Validate the input patch size.
        assert patchSize in [4, 8, 16], 'The pacth size should be 4, 8, or 16!'
        # Set the patch embedding sequence.
        self.patchEmbed = nn.Sequential(
            # Input size becomes half.
            nn.Conv2d(inChannel, hiddenSize, kernel_size = 7, stride = stride, padding = 3, bias = False),
            nn.BatchNorm2d(hiddenSize),
            nn.ReLU(inplace = True),
            nn.Conv2d(hiddenSize, hiddenSize, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(hiddenSize),
            nn.ReLU(inplace = True),
            nn.Conv2d(hiddenSize, hiddenSize, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(hiddenSize),
            nn.ReLU(inplace = True)
        )
        # Set the convolutional layer to project the output.
        self.projLayer = nn.Conv2d(hiddenSize, embedSize, kernel_size = patchSize // stride, stride = patchSize // stride)
    
    # Create the forward.
    def forward(self, x):
        # Compute the patch embedding. [B, C, H, W] -> [B, H/2, W/2, Hidden]
        x = self.patchEmbed(x)
        # Compute the output. [B, H/2, W/2, Hidden] -> [B, H', W', Embed]
        output = self.projLayer(x)
        # Return the output.
        return output

# Implement the downsampling.
class DownSample(nn.Module):
    '''
        Implement the downsampling.\n
        Params:\n
            - 'inputSize' is the size of the input data.
            - 'outputSize' is the size of the output data.
            - 'patchSize' is the size of each patch.
    '''

    # Create the constructor.
    def __init__(self, inputSize, outputSize, patchSize):
        # Create the super constructor.
        super(DownSample, self).__init__()
        # Set the downsampling layer.
        self.downSample = nn.Conv2d(inputSize, outputSize, kernel_size = patchSize, stride = patchSize)
    
    # Create the forward.
    def forward(self, x):
        # Compute the output.
        output = self.downSample(x)
        # Return the output.
        return output