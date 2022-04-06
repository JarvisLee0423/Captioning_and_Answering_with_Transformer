'''
    Copyright:      JarvisLee
    Date:           2022/01/15
    Filename:       DataPreprocessor.py
    Description:    Prepare the dataset.
'''

# Import the necessary library.
import torch
import torchvision.datasets as datasets
from torchvision.transforms import autoaugment, transforms
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from Utils.ParamsHandler import Handler

# Get the hyperparameters' handler.
Cfg = Handler.Parser(Handler.Generator(paramsDir = './Params.txt'))
# Set the tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encapsulate the MSCOCO dataloader's tools.
class MSCOCODataLoader():
    '''
        Encapsulate the MSCOCO dataloader's tools.\n
        Contains three parts:\n
            - 'GetData' gets the MSCOCO data.
            - 'Tokenize' tokenizes the captions.
            - 'Collate' loads each batch of the data.
            - 'DataLoader' loads all the data.
    '''

    # Get the MSCOCO data.
    def GetData(dataRoot = '.', annRoot = '.', cropSize = 224):
        '''
            This function is used to get the MSCOCO data.\n
            Params:\n
                - 'dataRoot' is the root of the MSCOCO data.
                - 'annRoot' is the root of the MSCOCO annotation.
                - 'cropSize' is the cropping size of the MSCOCO data.
        '''
        # Set the images' transformation.
        transform = transforms.Compose([
            transforms.RandomResizedCrop(cropSize),
            transforms.RandomHorizontalFlip(),
            autoaugment.RandAugment(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        # Get the MSCOCO datasets.
        dataset = datasets.CocoCaptions(root = dataRoot, annFile = annRoot, transform = transform)
        # Return the MSCOCO datasets.
        return dataset
    
    # Tokenize the captions.
    def Tokenize(captions, tokenizer):
        '''
            Tokenize the captions.\n
            Params:\n
                - 'captions' is the original captions.
                - 'tokenizer' is the tokenizer.
        '''
        # Tokenize the original captions.
        captionTokenDict = tokenizer.batch_encode_plus(
            captions,
            add_special_tokens = True,
            padding = 'longest',
            truncation = False,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        # Create the list to store the tokenized train captions.
        captionTrainTokens = []
        # Remove the '[SEP]' token.
        for each in captionTokenDict['input_ids']:
            temp = each.tolist()
            temp.remove(102)
            captionTrainTokens.append(torch.tensor([temp]))
        # Concatenate all the tokenized train captions.
        captionTrainTokens = torch.cat(captionTrainTokens, dim = 0)
        # Create the list to store the tokenized target captions.
        captionTarTokens = captionTokenDict['input_ids'][:, 1:]
        # Create the corresponding padding masks.
        padMasks = captionTokenDict['attention_mask'][:, 1:]
        # Create the corresponding segmentation masks.
        segMasks = captionTokenDict['token_type_ids'][:, 1:]
        # Convert the values in segmentation masks to be 1.
        segMasks = torch.ones(segMasks.shape, dtype = torch.long)
        # Return the tokenized captions, the padding mask, and the segmentation mask.
        return captionTrainTokens, captionTarTokens, padMasks, segMasks
    
    # Implement the collate function.
    def Collate(data):
        '''
            Implement the collate function.\n
            Params:\n
                - 'data' is the MSCOCO data.
        '''
        # Get the images and captions.
        images, captions = zip(*data)
        # Merge the images into the batch.
        images = torch.stack(images, dim = 0)
        # Create the list to store the captions' sentences.
        capSents = []
        # Create the list to store the training captions' token.
        capTrainTokens = []
        # Create the list to store the target captions' token.
        capTarTokens = []
        # Create the list to store the captions' mask.
        capMasks = []
        # Create the list to store the segmentations' mask.
        segMasks = []
        # Get all the data.
        for i in range(5):
            # Get the caption's sentence.
            capSent = []
            # Get the data.
            for sent in captions:
                # Get the sentence.
                capSent.append(sent[i])
            # Tokenize the sentence.
            capTrainToken, capTarToken, capMask, segMask = MSCOCODataLoader.Tokenize(captions = capSent, tokenizer = tokenizer)
            # Store the data.
            capSents.append(capSent)
            capTrainTokens.append(capTrainToken)
            capTarTokens.append(capTarToken)
            capMasks.append(capMask)
            segMasks.append(segMask)
        # Return the data.
        return images, capSents, capTrainTokens, capTarTokens, capMasks, segMasks
    
    # Load the data.
    def DataLoader(dataRoot = '.', annRoot = '.', batchSize = 32, cropSize = 224):
        '''
            Load the data.\n
            Params:\n
                - 'dataRoot' is the root of the MSCOCO data.
                - 'annRoot' is the root of the MSCOCO annotation.
                - 'batchSize' is the batch size of the MSCOCO data.
                - 'cropSize' is the cropping size of the MSCOCO data.
        '''
        # Get the training dataset.
        trainDataset = MSCOCODataLoader.GetData(dataRoot = f'{dataRoot}/Train/', annRoot = f'{annRoot}/Annotations/captions_train2014.json', cropSize = cropSize)
        # Get the validation dataset.
        valDataset = MSCOCODataLoader.GetData(dataRoot = f'{dataRoot}/Val/', annRoot = f'{annRoot}/Annotations/captions_val2014.json', cropSize = cropSize)
        # Get the training data.
        trainData = DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True, drop_last = False, collate_fn = MSCOCODataLoader.Collate)
        # Get the validation data.
        valData = DataLoader(dataset = valDataset, batch_size = batchSize, shuffle = False, drop_last = False, collate_fn = MSCOCODataLoader.Collate)
        # Initialize the vocabulary.
        vocab = dict()
        # Get the vocabulary.
        for key, value in tokenizer.vocab.items():
            vocab[value] = key
        # Convert the vocabulary into a list.
        vocab = list(vocab.values())
        # Return the data and vocabulary.
        return trainData, valData, vocab