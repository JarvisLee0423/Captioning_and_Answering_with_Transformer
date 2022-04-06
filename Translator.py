'''
    Copyright:      JarvisLee
    Data:           2022/03/14
    Filename:       Translator.py
    Description:    Implement a demo pipeline with the trained Captioning and Answering with Transformer (CATR) architecture.
'''

# Import the necessary library.
import os
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from PIL import Image
from Models.CATR import CATR
from Utils.DataPreprocessor import MSCOCODataLoader
from Utils.ParamsHandler import Handler

# Get the hyperparameters.
Cfg = Handler.Parser(Handler.Generator(paramsDir = './Params.txt'))
# Set the tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Fix the training devices and random seed.
if torch.cuda.is_available():
    np.random.seed(Cfg.seed)
    torch.cuda.manual_seed(Cfg.seed)
    if Cfg.GPUID > -1:
        torch.cuda.manual_seed(Cfg.seed)
    device = 'cuda'
else:
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)
    device = 'cpu'

# Get the path of the model.
modelDir = f'{Cfg.modelDir}/Demo.pt'
# Get the path of the test image.
imgDir = f'{Cfg.dataDir}/Test/'
# Set the hyperparameters for the captioning and answering with transformer architecture according to the Vision-Outlooker state.
if Cfg.volo == 1:
    stageSizes = [4, 4, 8, 2, 8]
    headSizes = [6, 12, 12, 12, 8]
    embedSizes = [192, 384, 384, 384, 384]
    mlpRatios = [3, 3, 3, 3, 3]
    path = f'{Cfg.voloDir}//VOLO-Type-1-224.pt'
elif Cfg.volo == 2:
    stageSizes = [6, 4, 10, 4, 8]
    headSizes = [8, 16, 16, 16, 8]
    embedSizes = [256, 512, 512, 512, 512]
    mlpRatios = [3, 3, 3, 3, 3]
    path = f'{Cfg.voloDir}//VOLO-Type-2-224.pt'
elif Cfg.volo == 3:
    stageSizes = [8, 8, 16, 4, 12]
    headSizes = [8, 16, 16, 16, 12]
    embedSizes = [256, 512, 512, 512, 512]
    mlpRatios = [3, 3, 3, 3, 3]
    path = f'{Cfg.voloDir}//VOLO-Type-3-224.pt'
elif Cfg.volo == 4:
    stageSizes = [8, 8, 16, 4, 12]
    headSizes = [12, 16, 16, 16, 12]
    embedSizes = [384, 768, 768, 768, 768]
    mlpRatios = [3, 3, 3, 3, 3]
    path = f'{Cfg.voloDir}//VOLO-Type-4-224.pt'
else:
    stageSizes = [12, 12, 20, 4, 16]
    headSizes = [12, 16, 16, 16, 16]
    embedSizes = [384, 768, 768, 768, 768]
    mlpRatios = [4, 4, 4, 4, 4]
    path = f'{Cfg.voloDir}//VOLO-Type-5-224.pt'
# Set the image transforms.
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]
)

# Get the dataset.
trainSet, valSet, vocab = MSCOCODataLoader.DataLoader(dataRoot = Cfg.dataDir, annRoot = Cfg.dataDir, batchSize = Cfg.batchSize, cropSize = Cfg.cropSize)
# Create the model.
model = CATR(vocabSize = len(vocab), stageSizes = stageSizes, headSizes = headSizes, mlpRatios = mlpRatios, embedSizes = embedSizes, embedHiddenSize = Cfg.embedHiddenSize, bertHiddenSize = Cfg.bertHiddenSize, attenDrop = Cfg.attenDrop, projDrop = Cfg.projDrop, voloTrain = Cfg.voloTrain, bert = Cfg.bert, path = path)
# Load the model.
model.load_state_dict(torch.load(modelDir, map_location = 'cpu'), strict = True)
# Send the model into the corresponding device.
model = model.to(device)
# Make the model to be the evaluation mode.
model = model.eval()

# Create the main function to execute the demo.
if __name__ == "__main__":
    # Read all the files from the image directory.
    files = os.listdir(imgDir)
    # Check all the files.
    for file in files:
        # Load the test image.
        demoImg = Image.open(f'{imgDir}/{file}').convert('RGB')
        # Preprocess the image.
        demoImg = transform(demoImg)
        demoImg = demoImg.unsqueeze(0)
        # Ensure not compute all the gradients.
        with torch.no_grad():
            # Get the translation.
            genCap = CATR.beamTranslator(model, demoImg, Cfg.beamSize, vocab, device, tokenizer, Cfg.bert)
            # Show the generated caption.
            print(f'The result of the beam search with beam size equals {Cfg.beamSize}: ', genCap)
        # Show the image.
        plt.imshow(Image.open(f'{imgDir}/{file}'))
        plt.title(genCap.split('[SEP]')[0])
        plt.show()