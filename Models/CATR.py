'''
    Copyright:      JarvisLee
    Date:           2022/01/18
    Filename:       CATR.py
    Description:    Generate the Captioning and Answering with Transformer (CATR) architecture.
'''

# Import the necessary library.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data.metrics as M
from Models.Modules import VisionEncoder, LinguisticEncoder, CaptionDecoder
from Utils.DataPreprocessor import MSCOCODataLoader

# Implement the captioning and answering with transformer architecture. (CATR)
class CATR(nn.Module):
    '''
        Implement the CATR.\n
        Params:\n
            - 'vocabSize' is the size of the vocabulary size.
            - 'stageSizes' is a list to show the number of blocks in each stage.
            - 'headSizes' is a list to show the size of the head of each block in each stage.
            - 'mlpRatios' is a list to show the size of the multi-layer perceptron's hidden size ratio of each block in each stage.
            - 'embedSizes' is a list to show the embedding size of each stage.
            - 'embedHiddenSize' is the hidden size of the patch embedding layer.
            - 'bertHiddenSize' is the dimension of the hidden state of the Bidirection Encoder Representation from Transformer.
            - 'attenDrop' is the drop rate of the attention dropout.
            - 'projDrop' is the drop rate of the linear dropout.
            - 'voloTrain' is the boolean to check whether compute the gradients for the Vision Outlooker.
            - 'bert' is the boolean to check whether applied the bert encoder.
            - 'path' is the path of the pre-trained checkpoints of the VOLO.
    '''

    # Create the constructor.
    def __init__(self, vocabSize, stageSizes, headSizes, mlpRatios, embedSizes, embedHiddenSize, bertHiddenSize, attenDrop = 0.0, projDrop = 0.0, voloTrain = True, bert = True, path = None):
        # Create the super constructor.
        super(CATR, self).__init__()
        # Get the BERT signal.
        self.bertSignal = bert
        # Create the vision encoder.
        self.visionEncoder = VisionEncoder(stageSizes = stageSizes[0:4], headSizes = headSizes[0:4], mlpRatios = mlpRatios[0:4], embedSizes = embedSizes[0:4], embedHiddenSize = embedHiddenSize, attenDrop = attenDrop, projDrop = projDrop, voloTrain = voloTrain, path = path)
        # Check whether applied the BERT encoder.
        if bert:
            # Create the linguistic encoder.
            self.linguisticEncoder = LinguisticEncoder(bertHiddenSize = bertHiddenSize)
        # Create the caption decoder.
        self.captionDecoder = CaptionDecoder(vocabSize = vocabSize, blockSize = stageSizes[4], bertHiddenSize = bertHiddenSize, hiddenSize = embedSizes[4], headSize = headSizes[4], mlpRatio = mlpRatios[4], attenDrop = attenDrop, projDrop = projDrop, word2vec = (bert == 0))
    
    # Compute the accuracy.
    def accuracy(predict, target, padMask):
        '''
            This function is used to get the accuracy between the prediction and the target.\n
            Params:\n
                - 'predict' is the prediction.
                - 'target' is the target.
                - 'padMask' is the padding mask of the prediction and target.
        '''
        # Get the predicted tokens.
        predTokens = torch.argmax(predict, dim = 2)
        # Get the accuracy tokens.
        accTokens = (predTokens == target)
        # Mask the accuracy tokens.
        accTokens = accTokens.masked_fill(padMask == 0, False)
        # Compute the accuracy
        accuracy = accTokens.sum().float() / padMask.sum().item()
        # Return the accuracy.
        return accuracy.item()
    
    # Compute the Bi-Lingual Evaluation Understudy scores. (BLEU)
    def BLEU(predict, targets, tarPadMasks, vocab):
        '''
            This function is used to get the four kinds of bi-lingual evaluation understudy scores between the prediction and target.\n
            Params:\n
                - 'predict' is the prediction.
                - 'targets' is the target.
                - 'tarPadMasks' is the padding mask of the target.
                - 'vocab' is the vocabulary list.
        '''
        # Get the predicted tokens.
        predTokens = torch.argmax(predict, dim = 2)
        # Get the prediction masks.
        predPadMask = (predTokens != vocab.index('[PAD]'))
        # Convert predicted and target tokens into source and references.
        source = []
        reference = []
        for i in range(predict.shape[0]):
            predSent = [vocab[idx] for idx in predTokens[i][0:(torch.sum(predPadMask[i]) - 1)]]
            source.append(predSent)
            tarSents = []
            for j in range(5):
                tarSent = [vocab[idx] for idx in targets[j][i][0:(torch.sum(tarPadMasks[j][i]) - 1)]]
                tarSents.append(tarSent)
            reference.append(tarSents)
        # Initialize the BLEU scores.
        BLEU1 = M.bleu_score(source, reference, max_n = 1, weights = [1] * 1)
        BLEU2 = M.bleu_score(source, reference, max_n = 2, weights = [1 / 2] * 2)
        BLEU3 = M.bleu_score(source, reference, max_n = 3, weights = [1 / 3] * 3)
        BLEU4 = M.bleu_score(source, reference, max_n = 4, weights = [1 / 4] * 4)
        # Return the BLEU scores
        return BLEU1, BLEU2, BLEU3, BLEU4

    # Apply the beam search to complete the translation.
    def beamTranslator(model, image, beamSize, vocab, device, tokenizer, bert, maxLen = 50, alpha = 0.7):
        # Initialize all the parameters.
        padIdx = vocab.index('[PAD]')
        startIdx = vocab.index('[CLS]')
        endIdx = vocab.index('[SEP]')
        # Initialize the start caption.
        startCap = torch.LongTensor([[startIdx]])
        # Initialize the start padding mask.
        startPadMask = (startCap != padIdx)
        # Initialize the start segmentation mask.
        startSegMask = torch.zeros(startCap.shape, dtype = torch.long)
        # Initialize the generated caption.
        genCap = torch.full((beamSize, maxLen), padIdx, dtype = torch.long).detach()
        genCap[:, 0] = startIdx
        # Initialize the length mapping.
        lenMap = torch.arange(1, maxLen + 1, dtype = torch.long).unsqueeze(0).to(device)
        # Generate the captions.
        with torch.no_grad():
            # Get the vision and linguistic encodes.
            vision = model.visionEncoder(image.to(device))
            # Check the signal of the BERT encoder.
            if bert:
                # Compute the linguistic features. [B, L] -> [B, L, C]
                linguistic = model.linguisticEncoder(startCap.to(device), startPadMask.to(device), startSegMask.to(device))
                # Compute the output.
                _, output = model.captionDecoder((vision.to(device), linguistic.to(device)), startPadMask.to(device))
            else:
                # Compute the output.
                _, output = model.captionDecoder((vision.to(device), startCap.to(device)), startPadMask.to(device))
            output = F.log_softmax(output, dim = -1)
            # Generate the first captions.
            kProbs, kIdx = output[:, -1, :].topk(beamSize)
            scores = kProbs.view(beamSize)
            genCap[:, 1] = kIdx[0]
            # Repeat the vision encodes.
            vision = vision.repeat(beamSize, 1, 1)
            # Start the remaining generation.
            for step in range(2, maxLen):
                # Get the new pad mask.
                padMask = (genCap[:, :step] != padIdx)
                # Get the new segmentation mask.
                segMask = torch.zeros(genCap[:, :step].shape, dtype = torch.long)
                # Check the signal of the BERT encoder.
                if bert:
                    # Compute the linguistic features. [B, L] -> [B, L, C]
                    linguistic = model.linguisticEncoder(genCap[:, :step].to(device), padMask.to(device), segMask.to(device))
                    # Compute the output.
                    _, output = model.captionDecoder((vision.to(device), linguistic.to(device)), padMask.to(device))
                else:
                    # Compute the output.
                    _, output = model.captionDecoder((vision.to(device), genCap[:, :step].to(device)), padMask.to(device))
                output = F.log_softmax(output, dim = -1)
                # Get the new top-k output.
                tempKProbs, tempKIdx = output[:, -1, :].topk(beamSize)
                # Get the new scores.
                scores = tempKProbs.view(beamSize, -1) + scores.view(beamSize, 1)
                scores, kTempKIdx = scores.view(-1).topk(beamSize)
                # Get the indices of the top-k in the table with the shape of beamSize times beamSize.
                rowIdx, columnIdx = kTempKIdx.cpu().numpy() // beamSize, kTempKIdx.cpu().numpy() % beamSize
                kIdx = tempKIdx[rowIdx, columnIdx]
                # Update the generated captions.
                genCap[:, :step] = genCap[rowIdx, :step]
                genCap[:, step] = kIdx
                # Check the location of the end token.
                endLocation = (genCap == endIdx).to(device)
                genLens, _ = lenMap.masked_fill(~endLocation, maxLen).min(1)
                # Check the terminate condition.
                if (endLocation.sum(1) > 0).sum(0).item() == beamSize:
                    # Get the translation index.
                    _, transIdx = scores.div(genLens.float() ** alpha).max(0)
                    transIdx = transIdx.item()
            # Compute the translations indcies if the for loop is not early terminated.
            if step == (maxLen - 1):
                # Get the translation index.
                _, transIdx = scores.div(genLens.float() ** alpha).max(0)
                transIdx = transIdx.item()
        # Get the tokens of the generated captions.
        genToken = genCap[transIdx][1:genLens[transIdx]].tolist()
        genSent = [vocab[idx] for idx in genToken]
        # Return the generated captions and their scores.
        return tokenizer.convert_tokens_to_string(genSent)

    # Apply the greedy method to translate the image.
    def greedyTranslator(model, image, vocab, device, tokenizer, bert):
        # Create the list to store the sentence.
        capSent = ['']
        # Create the list to store all the tokens.
        capToken = []
        # Tokenize the sentence.
        capTrain, capTar, padMask, segMask = MSCOCODataLoader.Tokenize(capSent, tokenizer)
        # Get the first vision and linguistic encodes.
        vision = model.visionEncoder(image.to(device))
        # Check the signal of the BERT encoder.
        if bert:
            # Compute the linguistic features. [B, L] -> [B, L, C]
            linguistic = model.linguisticEncoder(capTrain.to(device), padMask.to(device), segMask.to(device))
            # Compute the output.
            _, word = model.captionDecoder((vision.to(device), linguistic.to(device)), padMask.to(device))
        else:
            # Compute the output.
            _, word = model.captionDecoder((vision.to(device), capTrain.to(device)), padMask.to(device))
        word = torch.argmax(word[:, -1, :], dim = 1)
        word = vocab[word.squeeze()]
        # Store the new token.
        capToken.append(word)
        # Get the new sentence.
        capSent[0] = tokenizer.convert_tokens_to_string(capToken)
        # Start the generation.
        while word != '[SEP]':
            # Tokenize the sentence.
            capTrain, capTar, padMask, segMask = MSCOCODataLoader.Tokenize(capSent, tokenizer)
            # Check the signal of the BERT encoder.
            if bert:
                # Compute the linguistic features. [B, L] -> [B, L, C]
                linguistic = model.linguisticEncoder(capTrain.to(device), padMask.to(device), segMask.to(device))
                # Compute the output.
                _, word = model.captionDecoder((vision.to(device), linguistic.to(device)), padMask.to(device))
            else:
                # Compute the output.
                _, word = model.captionDecoder((vision.to(device), capTrain.to(device)), padMask.to(device))
            word = torch.argmax(word[:, -1, :], dim = 1)
            word = vocab[word.squeeze()]
            # Store the new token.
            capToken.append(word)
            # Get the new sentence.
            capSent[0] = tokenizer.convert_tokens_to_string(capToken)
        # Return the translation result.
        return capSent[0]
    
    # Create the forward.
    def forward(self, x, padMasks = None, segMasks = None):
        # Get the input data.
        images, captions = x
        # Compute the vision features. [B, C, H, W] -> [B, L, C] 
        visFeat = self.visionEncoder(images)
        # Check the signal of the BERT encoder.
        if self.bertSignal:
            # Compute the linguistic features. [B, L] -> [B, L, C]
            linFeat = self.linguisticEncoder(captions, padMasks, segMasks)
            # Compute the output.
            visFeat, linFeat = self.captionDecoder((visFeat, linFeat), padMasks)
        else:
            # Compute the output.
            visFeat, linFeat = self.captionDecoder((visFeat, captions), padMasks)
        # Return the vision and output.
        return visFeat, linFeat