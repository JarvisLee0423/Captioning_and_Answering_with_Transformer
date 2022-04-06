'''
    Copyright:      JarvisLee
    Date:           2022/01/20
    Filename:       Trainer.py
    Description:    Implement a trainer to train the captioning and answering with transformer architecture. (CATR)
'''

# Import the necessary library.
import os
import time
import pynvml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Models.CATR import CATR
from Models.Modules import NoamScheduler
from Utils.DataPreprocessor import MSCOCODataLoader
from Utils.InfoLogger import Logger
from Utils.ParamsHandler import Handler

# Get the hyperparameters.
Cfg = Handler.Parser(Handler.Generator(paramsDir = './Params.txt'))
# Get the current time.
currentTime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

# Check the directtory.
if not os.path.exists(Cfg.modelDir):
    os.mkdir(Cfg.modelDir)
if not os.path.exists(f'{Cfg.modelDir}//{currentTime}'):
    os.mkdir(f'{Cfg.modelDir}//{currentTime}')
if not os.path.exists(Cfg.logDir):
    os.mkdir(Cfg.logDir)
if not os.path.exists(f'{Cfg.logDir}//{currentTime}'):
    os.mkdir(f'{Cfg.logDir}//{currentTime}')

# Fix the training devices and random seed.
if torch.cuda.is_available():
    np.random.seed(Cfg.seed)
    torch.cuda.manual_seed(Cfg.seed)
    if Cfg.GPUID > -1:
        torch.cuda.manual_seed(Cfg.seed)
        # Get the GPU logger.
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(Cfg.GPUID)
    device = 'cuda'
else:
    np.random.seed(Cfg.seed)
    torch.manual_seed(Cfg.seed)
    device = 'cpu'

# Indicate whether the Vision-Outlooker state is valid or not.
assert Cfg.volo >= 1 and Cfg.volo <= 5, 'Please set a valid Vision-Outlooker state! (1 <= volo <= 5)'
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

# Implement the training pipeline.
class Trainer():
    '''
        Implement the training pipeline.\n
        Contains two parts:\n
            - 'trainer' is used to do the training.
            - 'evaluator' is used to do the evaluating.
    '''

    # Train the model.
    def trainer(model, loss, optim, trainData, valData, vocab, dataID, epoch, epochs, device, scheduler = None, eval = True):
        '''
            Train the model.\n
            Params:\n
                - 'model' is the captioning and answering with transformer architecture.
                - 'loss' is the loss function.
                - 'optim' is the optimizer.
                - 'trainData' is the training data.
                - 'valData' is the validation data.
                - 'vocab' is the vocabulary.
                - 'dataID' is the index of the data.
                - 'epoch' is the current training epoch.
                - 'epochs' is the total training epochs.
                - 'device' is the device setting.
                - 'scheduler' is the learning rate scheduler.
                - 'eval' is the boolean value to indicate whether applying the validation during the training.
        '''
        # Initialize the lists to store the training loss, accuracy and bi-lingual evaluation understudy scores.
        trainLoss = []
        trainAcc = []
        trainBLEU1 = []
        trainBLEU2 = []
        trainBLEU3 = []
        trainBLEU4 = []
        # Initialize the learning rate list.
        lrList = []
        # Set the training loading bar.
        with tqdm(total = len(trainData), desc = f'Epoch {epoch + 1}/{epochs}', unit = 'batch', dynamic_ncols = True) as pbars:
            # Get the training data.
            for i, (images, capSents, capTrainTokens, capTarTokens, capMasks, segMasks) in enumerate(trainData):
                # Sent the data into corresponding device.
                image = images.to(device)
                capTrainToken = capTrainTokens[dataID].to(device)
                capTarToken = capTarTokens[dataID].to(device)
                capMask = capMasks[dataID].to(device)
                segMask = segMasks[dataID].to(device)
                # Compute the prediction.
                vision, linguistic = model((image, capTrainToken), capMask, segMask)
                # Convert the shape of the prediction and target.
                B, L, _ = linguistic.shape
                predictions = linguistic.reshape(B * L, -1)
                targets = capTarToken.reshape(B * L)
                # Compute the loss.
                cost = loss(predictions, targets)
                # Store the loss.
                trainLoss.append(cost.item())
                # Check whether apply the inner learning rate scheduler.
                if scheduler is not None:
                    scheduler.step()
                # Store the learning rate.
                lrList.append(optim.state_dict()['param_groups'][0]['lr'])
                # Clear the previous gradient.
                optim.zero_grad()
                # Compute the backward.
                cost.backward()
                # Update the parameters.
                optim.step()
                # Compute the accuracy.
                accuracy = CATR.accuracy(predict = linguistic, target = capTarToken, padMask = capMask)
                # Store the accuracy.
                trainAcc.append(accuracy)
                # Compute the bi-lingual evaluation understudy scores.
                BLEU1, BLEU2, BLEU3, BLEU4 = CATR.BLEU(predict = linguistic, targets = capTarTokens, tarPadMasks = capMasks, vocab = vocab)
                # Store the bi-lingual evaluation understudy scores.
                trainBLEU1.append(BLEU1)
                trainBLEU2.append(BLEU2)
                trainBLEU3.append(BLEU3)
                trainBLEU4.append(BLEU4)
                # Update the loading bar.
                pbars.update(1)
                # Update the training information.
                pbars.set_postfix_str(' - Train Loss %.4f - Train Acc %.4f - Train BLEUs [%.4f, %.4f, %.4f, %.4f]' % (np.mean(trainLoss), np.mean(trainAcc), np.mean(trainBLEU1), np.mean(trainBLEU2), np.mean(trainBLEU3), np.mean(trainBLEU4)))
        # Close the loading bar.
        pbars.close()
        # Check whether do the evaluation.
        if eval == True:
            # Output the hint for evaluation.
            print('Evaluating...', end = ' ')
            # Evaluating the model.
            evalLoss, evalAcc, evalBLEU1, evalBLEU2, evalBLEU3, evalBLEU4 = Trainer.evaluator(model.eval(), loss, valData, vocab, dataID, device)
            # Output the evaluating results.
            print('- Eval Loss %.4f - Eval Acc %.4f - Eval BLEUs [%.4f, %.4f, %.4f, %.4f]' % (evalLoss, evalAcc, evalBLEU1, evalBLEU2, evalBLEU3, evalBLEU4))
            # Return the training results.
            return model.train(), lrList, np.mean(trainLoss), np.mean(trainAcc), [np.mean(trainBLEU1), np.mean(trainBLEU2), np.mean(trainBLEU3), np.mean(trainBLEU4)], evalLoss, evalAcc, [evalBLEU1, evalBLEU2, evalBLEU3, evalBLEU4]
        # Return the training results.
        return model.train(), lrList, np.mean(trainLoss), np.mean(trainAcc), [np.mean(trainBLEU1), np.mean(trainBLEU2), np.mean(trainBLEU3), np.mean(trainBLEU4)], None, None, [None, None, None, None]
    
    # Evaluate the model.
    def evaluator(model, loss, valData, vocab, dataID, device):
        '''
            Evaluate the model.\n
            Params:\n
                - 'model' is the captioning and answering with transformer architecture.
                - 'loss' is the lossfunction.
                - 'valData' is the validation data.
                - 'vocab' is the vocabulary.
                - 'dataID' is the index of the data.
                - 'device' is the device setting.
        '''
        # Initialize the lists to store the evaluating loss, accuracy and bi-lingual evaluation understudy scores.
        evalLoss = []
        evalAcc = []
        evalBLEU1 = []
        evalBLEU2 = []
        evalBLEU3 = []
        evalBLEU4 = []
        # Get the evaluating data.
        with torch.no_grad():
            for i, (images, capSents, capTrainTokens, capTarTokens, capMasks, segMasks) in enumerate(valData):
                # Sent the data into corresponding device.
                image = images.to(device)
                capTrainToken = capTrainTokens[dataID].to(device)
                capTarToken = capTarTokens[dataID].to(device)
                capMask = capMasks[dataID].to(device)
                segMask = segMasks[dataID].to(device)
                # Compute the prediction.
                vision, linguistic = model((image, capTrainToken), capMask, segMask)
                # Convert the shape of the prediction and target.
                B, L, _ = linguistic.shape
                predictions = linguistic.reshape(B * L, -1)
                targets = capTarToken.reshape(B * L)
                # Compute the loss.
                cost = loss(predictions, targets)
                # Store the loss.
                evalLoss.append(cost.item())
                # Compute the accuracy.
                accuracy = CATR.accuracy(predict = linguistic, target = capTarToken, padMask = capMask)
                # Store the accuracy.
                evalAcc.append(accuracy)
                # Compute the bi-lingual evaluation understudy scores.
                BLEU1, BLEU2, BLEU3, BLEU4 = CATR.BLEU(predict = linguistic, targets = capTarTokens, tarPadMasks = capMasks, vocab = vocab)
                # Store the bi-lingual evaluation understudy scores.
                evalBLEU1.append(BLEU1)
                evalBLEU2.append(BLEU2)
                evalBLEU3.append(BLEU3)
                evalBLEU4.append(BLEU4)
            # Return the evaluating results.
            return np.mean(evalLoss), np.mean(evalAcc), np.mean(evalBLEU1), np.mean(evalBLEU2), np.mean(evalBLEU3), np.mean(evalBLEU4)

# Train the model.
if __name__ == "__main__":
    # Initialize the visdom server.
    vis = Logger.VisConfigurator(currentTime = currentTime, visName = f'{currentTime}')
    # Initialize the logger.
    logger = Logger.LogConfigurator(logDir = f'{Cfg.logDir}//{currentTime}//', filename = f'{currentTime}.txt')
    # Log the hyperparameters.
    logger.info('\n' + Handler.Displayer(Cfg))
    # Get the dataset.
    trainSet, valSet, vocab = MSCOCODataLoader.DataLoader(dataRoot = Cfg.dataDir, annRoot = Cfg.dataDir, batchSize = Cfg.batchSize, cropSize = Cfg.cropSize)
    # Create the model.
    model = CATR(vocabSize = len(vocab), stageSizes = stageSizes, headSizes = headSizes, mlpRatios = mlpRatios, embedSizes = embedSizes, embedHiddenSize = Cfg.embedHiddenSize, bertHiddenSize = Cfg.bertHiddenSize, attenDrop = Cfg.attenDrop, projDrop = Cfg.projDrop, voloTrain = Cfg.voloTrain, bert = Cfg.bert, path = path)
    # Send the model to the corresponding device.
    model = model.to(device)
    # Create the loss function.
    loss = nn.CrossEntropyLoss(ignore_index = 0, reduction = 'mean', label_smoothing = Cfg.smoothing)
    # Create the optimizer.
    optimizer = optim.Adam(model.parameters(), lr = Cfg.learningRate, betas = (Cfg.beta1, Cfg.beta2), eps = Cfg.epsilon, weight_decay = Cfg.weightDecay)
    # Create the learning rate scheduler.
    scheduler = NoamScheduler(optimizer, Cfg.warmUp, embedSizes[-1])
    # Create the learning rates storer.
    lrs = []
    # Train the model.
    for epoch in range(Cfg.epochs):
        # Compute the data index for each epoch.
        dataID = (epoch + 5) % 5
        # Train the model.
        model, lrList, trainLoss, trainAcc, trainBLEUs, evalLoss, evalAcc, evalBLEUs = Trainer.trainer(model = model, loss = loss, optim = optimizer, trainData = trainSet, valData = valSet, vocab = vocab, dataID = dataID, epoch = epoch, epochs = Cfg.epochs, device = device, scheduler = scheduler, eval = True)
        # Get the current learning rates.
        lrs.extend(lrList)
        # Store the learning rates.
        with open(f'{Cfg.logDir}//{currentTime}//learningRates.txt', 'w') as file:
            file.write(str(lrs))
        # Log the training results.
        if Cfg.GPUID > -1:
            # Compute the memory usage.
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024
            # Output the memory usage.
            print('- Memory %.4f/%.4f MB' % (memory, pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
        else:
            print(' ')
        if evalLoss == None:
            if Cfg.GPUID > -1:
                logger.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] - BLEUs [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f] || Memory: [%.4f/%.4f] MB' % (epoch + 1, Cfg.epochs, trainLoss, trainAcc, trainBLEUs[0], trainBLEUs[1], trainBLEUs[2], trainBLEUs[3], optimizer.state_dict()['param_groups'][0]['lr'], memory, pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
            else:
                logger.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] - BLEUs [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f]' % (epoch + 1, Cfg.epochs, trainLoss, trainAcc, trainBLEUs[0], trainBLEUs[1], trainBLEUs[2], trainBLEUs[3], optimizer.state_dict()['param_groups'][0]['lr']))
        else:
            if Cfg.GPUID > -1:
                logger.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] - BLEUs [%.4f, %.4f, %.4f, %.4f] || Evaluating: Loss [%.4f] - Acc [%.4f] - BLEUs [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f] || Memory: [%.4f/%.4f] MB' % (epoch + 1, Cfg.epochs, trainLoss, trainAcc, trainBLEUs[0], trainBLEUs[1], trainBLEUs[2], trainBLEUs[3], evalLoss, evalAcc, evalBLEUs[0], evalBLEUs[1], evalBLEUs[2], evalBLEUs[3], optimizer.state_dict()['param_groups'][0]['lr'], memory, pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024))
            else:
                logger.info('Epoch [%d/%d] -> Training: Loss [%.4f] - Acc [%.4f] - BLEUs [%.4f, %.4f, %.4f, %.4f] || Evaluating: Loss [%.4f] - Acc [%.4f] - BLEUs [%.4f, %.4f, %.4f, %.4f] || lr: [%.10f]' % (epoch + 1, Cfg.epochs, trainLoss, trainAcc, trainBLEUs[0], trainBLEUs[1], trainBLEUs[2], trainBLEUs[3], evalLoss, evalAcc, evalBLEUs[0], evalBLEUs[1], evalBLEUs[2], evalBLEUs[3], optimizer.state_dict()['param_groups'][0]['lr']))
        Logger.VisDrawer(vis = vis, epoch = epoch + 1, trainLoss = trainLoss, evalLoss = evalLoss, trainAcc = trainAcc, evalAcc = evalAcc, trainBLEUv1 = trainBLEUs[0], evalBLEUv1 = evalBLEUs[0], trainBLEUv2 = trainBLEUs[1], evalBLEUv2 = evalBLEUs[1], trainBLEUv3 = trainBLEUs[2], evalBLEUv3 = evalBLEUs[2], trainBLEUv4 = trainBLEUs[3], evalBLEUv4 = evalBLEUs[3])
        # Save the model.
        torch.save(model.state_dict(), f'{Cfg.modelDir}//{currentTime}//{currentTime}-{epoch + 1}.pt')
        logger.info('Model Saved')
    # Close the visdom server.
    Logger.VisSaver(vis, visName = f'{currentTime}')