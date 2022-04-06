'''
    Copyright:      JarvisLee
    Date:           2022/01/17
    Filename:       Modules.py
    Description:    Implement the necessary modules of the Captioning and Answering with Transformer.
'''

# Import the necessary library.
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from transformers import BertModel
from Models.Layers import OutlookAttention, MLP, SelfAttention, PatchEmbed, DownSample, PosEmbed

# Implement the noam learning rate scheduler.
class NoamScheduler():
    '''
        Implement the noam learning rate scheduler.\n
        Params:\n
            - 'optimizer' is the optimization method of the model training.
            - 'warmUp' is the total learning rate warm up steps.
            - 'hiddenSize' is the size of the input hidden data.
    '''

    # Create the constructor.
    def __init__(self, optimizer, warmUp, hiddenSize):
        # Set the memeber variables.
        self.optim = optimizer
        self.warmUp = warmUp
        self.hiddenSize = hiddenSize
        self.stepNum = 0
    
    # Create the step.
    def step(self):
        # Get the current step.
        self.stepNum = self.stepNum + 1
        # Compute the current learning rate.
        clr = (self.hiddenSize ** (-0.5)) * min(self.stepNum ** (-0.5), self.stepNum * self.warmUp ** (-1.5))
        # Update the learning rate in the optimizer.
        for params in self.optim.param_groups:
            params['lr'] = clr

# Implement the Outlooker.
class OutlookerBlock(nn.Module):
    '''
        Implement the outlooker block.\n
        Params:\n
            - 'hiddenSize' is the size of the input hidden data.
            - 'headSize' is the size of the head of the outlook-attention.
            - 'kernelSize' is the size of the attention window of the outlook-attention.
            - 'padding' is the padding of the attention window of the outlook-attention.
            - 'stride' is the stride of the attention window of the outlook-attention.
            - 'mlpRatio' is the multiple between the hidden size and the multi-layer perception's hidden size.
            - 'attenDrop' is the drop rate of the attention dropout.
            - 'projDrop' is the drop rate of the linear dropout.
    '''

    # Create the constructor.
    def __init__(self, hiddenSize, headSize, kernelSize, padding, stride = 1, mlpRatio = 3, attenDrop = 0.0, projDrop = 0.0):
        # Create the super constructor.
        super(OutlookerBlock, self).__init__()
        # Set the first layer normalization.
        self.layerNorm1 = nn.LayerNorm(hiddenSize)
        # Set the outlook-attention.
        self.atten = OutlookAttention(hiddenSize = hiddenSize, headSize = headSize, kernelSize = kernelSize, padding = padding, stride = stride, attenDrop = attenDrop, projDrop = projDrop)
        # Set the second layer normalization.
        self.layerNorm2 = nn.LayerNorm(hiddenSize)
        # Set the multi-layer perceptron.
        self.mlp = MLP(inputSize = hiddenSize, hiddenSize = int(hiddenSize * mlpRatio), projDrop = projDrop)

    # Create the forward.
    def forward(self, x):
        # Compute the output.
        x = x + self.atten(self.layerNorm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        output = x + self.mlp(self.layerNorm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        # Return the output.
        return output

# Implement the Transformer.
class TransformerBlock(nn.Module):
    '''
        Implement the transformer block.\n
        Params:\n
            - 'hiddenSize' is the size of the input hidden data.
            - 'bertHiddenSize' is the dimension of the hidden state of the Bidirection Encoder Representations from Transformer.
            - 'headSize' is the size of the head of the self-attention.
            - 'mlpRatio' is the multiple between the hidden size and the multi-layer perception's hidden size.
            - 'attenDrop' is the drop rate of the attention dropout.
            - 'projDrop' is the drop rate of the linear dropout.
            - 'decoder' is the boolean to indicate whether the transformer block is part of the decoder.
            - 'word2vec' is the boolean to indicate whether apply the traditional word embedding.
    '''

    # Create the constructor.
    def __init__(self, hiddenSize, bertHiddenSize, headSize, mlpRatio, attenDrop = 0.0, projDrop = 0.0, decoder = False, word2vec = False):
        # Create the super constructor.
        super(TransformerBlock, self).__init__()
        # Check whether the transformer is the decoder.
        if decoder:
            # Set the state of the transformer.
            self.state = 'decoder'
            # Set the masked-self-attention.
            self.maskAtten = SelfAttention(hiddenSize = hiddenSize, bertHiddenSize = bertHiddenSize, headSize = headSize, attenDrop = attenDrop, projDrop = projDrop, maskAtten = True, word2vec = word2vec)
            # Set the cross-self-attention.
            self.crossAtten = SelfAttention(hiddenSize = hiddenSize, bertHiddenSize = bertHiddenSize, headSize = headSize, attenDrop = attenDrop, projDrop = projDrop, crossAtten = True, word2vec = word2vec)
            # Check whether applied the traditional word embedding.
            if word2vec:
                # Set the layer normalizations.
                self.layerNorm1 = nn.LayerNorm(hiddenSize)
                self.layerNorm2 = nn.LayerNorm(hiddenSize)
                self.layerNorm3 = nn.LayerNorm(hiddenSize)
                # Set the multi-layer perceptron.
                self.mlp = MLP(inputSize = hiddenSize, hiddenSize = int(hiddenSize * mlpRatio), projDrop = projDrop, decoder = decoder)
            else:
                # Set the layer normalizations.
                self.layerNorm1 = nn.LayerNorm(bertHiddenSize)
                self.layerNorm2 = nn.LayerNorm(bertHiddenSize)
                self.layerNorm3 = nn.LayerNorm(bertHiddenSize)
                # Set the multi-layer perceptron.
                self.mlp = MLP(inputSize = bertHiddenSize, hiddenSize = int(hiddenSize * mlpRatio), projDrop = projDrop, decoder = decoder)
        # Check whether the transformer is the encoder.
        else:
            # Set the state of the transformer.
            self.state = 'encoder'
            # Set the first layer normalization.
            self.layerNorm1 = nn.LayerNorm(hiddenSize)
            self.atten = SelfAttention(hiddenSize = hiddenSize, bertHiddenSize = bertHiddenSize, headSize = headSize, attenDrop = attenDrop)
            # Set the second layer normalization.
            self.layerNorm2 = nn.LayerNorm(hiddenSize)
            # Set the multi-layer perceptron.
            self.mlp = MLP(inputSize = hiddenSize, hiddenSize = int(hiddenSize * mlpRatio))
    
    # Create the forward.
    def forward(self, x, padMask = None, attenMask = None):
        # Compute the output.
        if self.state == 'decoder':
            # Check whether input the cross data.
            assert len(x) == 2, 'Please input a tuple which contains the vision and linguistic features!'
            # Get the input data.
            vision, linguistic = x
            linguistic = linguistic + self.layerNorm1(self.maskAtten(linguistic, padMask, attenMask))
            output = linguistic + self.layerNorm2(self.crossAtten((vision, linguistic), padMask))
            output = output + self.layerNorm3(self.mlp(output))
            # Return the output
            return vision, output
        else:
            x = x + self.atten(self.layerNorm1(x.permute(0, 2, 3, 1)).permute(0, 3, 2, 1))
            output = x + self.mlp(self.layerNorm2(x.permute(0, 2, 3, 1)).permute(0, 3, 2, 1))
            # Return the output
            return output

# Implement the Vision-Outlooker.
class VOLO(nn.Module):
    '''
        Implement the Vision-Outlooker.\n
        Params:\n
            - 'stageSize' is a list to show the number of blocks in each stage.
            - 'headSize' is a list to show the size of the head of each each block in each stage.
            - 'mlpRatio' is a list to show the size of the multi-layer perceptron's hidden size ratio of each block in each stage.
            - 'embedSize' is a list to show the embedding size of each stage.
            - 'inputSize' is the size of the input data.
            - 'inChannel' is the number of the channels of the input data.
            - 'kernelSize' is the kernel size of the outlook-attention.
            - 'stride' is the stride of the outlook-attention.
            - 'padding' is the padding of the outlook-attention.
            - 'patchSize' is the size of each patch.
            - 'embedHiddenSize' is the size of the input data of the patch embedding.
            - 'bertHiddenSize' is the dimension of the hidden state of the Bidirection Encoder Representations from Transformer.
            - 'dropRate' is the dropout rate of the positional embedding.
            - 'attenDrop' is the drop rate of the attention dropout.
            - 'projDrop' is the drop rate of the linear dropout.
    '''

    # Create the constructor.
    def __init__(self, stageSize, headSize, mlpRatio, embedSize, inputSize = 224, inChannel = 3, kernelSize = 3, stride = 2, padding = 1, patchSize = 8, embedHiddenSize = 64, bertHiddenSize = 768, dropRate = 0.0, attenDrop = 0.0, projDrop = 0.0):
        # Create the super constructor.
        super(VOLO, self).__init__()
        # Set the patch embedding layer.
        self.patchEmbed = PatchEmbed(inChannel = inChannel, hiddenSize = embedHiddenSize, embedSize = embedSize[0], stride = 2)
        # Initialize the positional embedding. ('2' is for that the patch embedding downsample the original input 2x.)
        self.posEmbed = nn.Parameter(torch.zeros(1, inputSize // patchSize // 2, inputSize // patchSize // 2, embedSize[-1]))
        # Set the dropout for the positional embedding.
        self.posDrop = nn.Dropout(p = dropRate)
        # Set the main stages.
        stages = []
        for i in range(len(stageSize)):
            # Set the stage-1 to be the outlooker-block.
            if i == 0:
                blocks = []
                for _ in range(stageSize[i]):
                    blocks.append(
                        OutlookerBlock(hiddenSize = embedSize[i], headSize = headSize[i], kernelSize = kernelSize, padding = padding, stride = stride, mlpRatio = mlpRatio[i], attenDrop = attenDrop, projDrop = projDrop)
                    )
                stage = nn.Sequential(*blocks)
                stages.append(stage)
                stages.append(DownSample(inputSize = embedSize[i], outputSize = embedSize[i + 1], patchSize  = 2))
            # Set the remaining stages to be the transformer-block.
            else:
                blocks = []
                for _ in range(stageSize[i]):
                    blocks.append(
                        TransformerBlock(hiddenSize = embedSize[i], bertHiddenSize = bertHiddenSize, headSize = headSize[i], mlpRatio = mlpRatio[i], attenDrop = attenDrop, projDrop = projDrop, decoder = False)
                    )
                stage = nn.Sequential(*blocks)
                stages.append(stage)
        # Convert the stage list into the modules.
        self.volo = nn.ModuleList(stages)
        # Apply the weights initialization.
        self.apply(self._init_weights)

    # Initialize the weights.
    def _init_weights(self, module):
        # Initialize the positional embedding.
        trunc_normal_(self.posEmbed, std = 0.02)
        # Initialize the weights of the linear layer.
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std = 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # Initialize the weights of the layer normalization.
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    # Indicate the weight which dose not need to do the weight decay.
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'posEmbed'}

    # Create the forward.
    def forward(self, x):
        # Compute the patch embedding.
        x = self.patchEmbed(x)
        # Compute the vision-outlooker.
        for i, stage in enumerate(self.volo):
            # Compute the positional embedding after the first stage.
            if i == 2:
                x = x + self.posEmbed.permute(0, 3, 1, 2)
                x = self.posDrop(x)
            x = stage(x)
        # Change the shape of the output.
        B, C, _, _ = x.shape
        output = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        # Return the extracted vision features.
        return output

# Implement the vision encoder.
class VisionEncoder(nn.Module):
    '''
        Implement the vision encoder based on vision outlooker (VOLO).\n
        Params:\n
            - 'stageSizes' is a list to show the number of blocks in each stage.
            - 'headSizes' is a list to show the size of the head of each block in each stage.
            - 'mlpRatios' is a list to show the size of the multi-layer perceptron's hidden size ratio of each block in each stage.
            - 'embedSizes' is a list to show the embedding size of each stage.
            - 'embedHiddenSize' is the hidden size of the patch embedding layer.
            - 'attenDrop' is the drop rate of the attention dropout.
            - 'projDrop' is the drop rate of the linear dropout.
            - 'voloTrain' is the boolean to check whether compute the gradients for the Vision Outlooker.
            - 'path' is the path of the pre-trained checkpoints of the VOLO.
    '''

    # Create the constructor.
    def __init__(self, stageSizes, headSizes, mlpRatios, embedSizes, embedHiddenSize, attenDrop = 0.0, projDrop = 0.0, voloTrain = True, path = None):
        # Create the super constructor.
        super(VisionEncoder, self).__init__()
        # Get the volo train signals.
        self.voloTrain = voloTrain
        # Create the vision outlooker.
        self.volo = VOLO(stageSize = stageSizes, headSize = headSizes, mlpRatio = mlpRatios, embedSize = embedSizes, embedHiddenSize = embedHiddenSize, attenDrop = attenDrop, projDrop = projDrop)
        # Load the pre-trained weight.
        if path is not None:
            # Give the hint.
            print('VOLO Params Loading...', end = ' ')
            # Load the pre-trained checkpoints of the vision-outlooker.
            self.volo.load_state_dict(torch.load(path, map_location = 'cpu'), strict = True)
            # Give the hint.
            print('Done')
        # Check whether compute the gradients for the Vision Outlooker.
        if not voloTrain:
            for param in self.volo.parameters():
                param.requires_grad = False
    
    # Create the forward.
    def forward(self, images):
        # Check whether train the volo.
        if not self.voloTrain:
            # Ensure not compute the gradients of the volo.
            with torch.no_grad():
                # Get the vision features.
                features = self.volo(images)
        else:
            # Get the vision featues.
            features = self.volo(images)
        # Return the vision features.
        return features

# Implement the linguistic encoder.
class LinguisticEncoder(nn.Module):
    '''
        Implement the linguistic encoder based on Bidirectional Encoder Representation from Transformer.\n
        Params:\n
            - 'bertHiddenSize' is the dimension of the hidden state of the Bidirection Encoder Representation from Transformer.
    '''

    # Create the constructor.
    def __init__(self, bertHiddenSize):
        # Create the super constructor.
        super(LinguisticEncoder, self).__init__()
        # Check the size of the hidden state of the Bidirectional Encoder Representation from Transformer.
        assert bertHiddenSize == 768 or bertHiddenSize == 3072, 'Please input the accepted size of the hidden state of the BERT, which is 768 or 3072!'
        # Get the size of the hidden state of the Bidirectional Encoder Representation from Transformer.
        self.bertHiddenSize = bertHiddenSize
        # Create the Bidirectional Encoder Representation from Transformer.
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        # Send the Bidrectional Encoder Representation from Transformer into evaluation mode.
        self.bert = self.bert.eval()
        # Set all parameters' required gradient state in the bidirectional encoder representation with transformer become 'False'.
        for param in self.bert.parameters():
            param.requires_grad = False
    
    # Create the forward.
    def forward(self, captions, padMasks, segMasks):
        # Ensure not computer the gradients of the bert.
        with torch.no_grad():
            # Get the linguistic hiddens.
            hiddens = self.bert(captions, padMasks, segMasks)
            # Get the linguistic embedding.
            embedding = hiddens[2]
        # Concatenate the embeddings from all the layers of the bert.
        embedding = torch.stack(embedding, dim = 0).permute(1, 2, 0, 3)
        # Get the linguistic features.
        if self.bertHiddenSize == 768:
            features = torch.sum(embedding[:, :, -4:, :], dim = 2)
        else:
            features = torch.cat([embedding[:,:,-1,:], embedding[:,:,-2,:], embedding[:,:,-3,:], embedding[:,:,-4,:]], dim = 2)
        # Return the linguistic features.
        return features

# Implement the caption decoder.
class CaptionDecoder(nn.Module):
    '''
        Implement the caption decoder.\n
        Params:\n
            - 'vocabSize' is the size of the vocabulary size.
            - 'blockSize' is the size of the transformer block in the caption decoder.
            - 'bertHiddenSize' is the dimension of the hidden state of the Bidirection Encoder Representation from Transformer.
            - 'hiddenSize' is the size of the input hidden data.
            - 'headSize' is the size of the head of the self-attention.
            - 'mlpRatios' is the size of the multi-layer perceptron's hidden size ratio of the self-attention.
            - 'attenDrop' is the drop rate of the attention dropout.
            - 'projDrop' is the drop rate of the linear dropout.
            - 'word2vec' is the boolean to indicate whether apply the traditional word embedding.
    '''

    # Create the constructor.
    def __init__(self, vocabSize, blockSize, bertHiddenSize, hiddenSize, headSize, mlpRatio, attenDrop = 0.0, projDrop = 0.0, word2vec = False):
        # Create the super constructor.
        super(CaptionDecoder, self).__init__()
        # Get the embedding signal.
        self.embedSignal = word2vec
        # Check whether applied the word embedding.
        if word2vec:
            # Give the hint about the linguistic encoder.
            print('Word Embedding has been employed.')
            # Create the linguistic encoder.
            self.wordEmbed = nn.Embedding(num_embeddings = vocabSize, embedding_dim = hiddenSize)
            # Create the output layer.
            self.output = nn.Linear(hiddenSize, vocabSize)
        else:
            # Give the hine about the linguistic encoder.
            print('BERT Embedding has been employed.')
            # Create the output layer.
            self.output = nn.Linear(bertHiddenSize, vocabSize)
        # Create the positional embedding.
        self.posEmbed = PosEmbed()
        # Set the decoder.
        self.decoder = []
        for _ in range(blockSize):
            self.decoder.append(
                TransformerBlock(hiddenSize = hiddenSize, bertHiddenSize = bertHiddenSize, headSize = headSize, mlpRatio = mlpRatio, attenDrop = attenDrop, projDrop = projDrop, decoder = True, word2vec = word2vec)
            )
        self.decoder = nn.ModuleList(self.decoder)
        # Apply the weights initialization.
        self.apply(self._init_weights)
    
    # Initialize the weights.
    def _init_weights(self, module):
        # Initialize the weights of the linear layer.
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # Initialize the weights of the layer normalization.
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        # Initialize the weights of the embedding layer.
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
    
    # Create the forward.
    def forward(self, x, padMasks = None):
        # Get the input.
        vision, linguistic = x
        # Check whether applied the word embedding.
        if self.embedSignal:
            # Compute the linguistci features.
            linguistic = self.wordEmbed(linguistic)
        # Get the attention mask. [1, 1, L, L]
        attenMasks = SelfAttention.getAttenMask(linguistic)
        # Get the positional embedding of the linguistic features. [1, L, C]
        posEmbeds = self.posEmbed(linguistic)
        # Combine the linguistic features and the positional embedding. [B, L, C] -> [B, L, C]
        linguistic = linguistic + posEmbeds
        # Compute the caption-decoder.
        for block in self.decoder:
            vision, linguistic = block((vision, linguistic), padMasks, attenMasks)
        # Get the output. [B, L, C] -> [B, L, 30522]
        linguistic = self.output(linguistic)
        # Return the output.
        return vision, linguistic