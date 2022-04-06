'''
    Copyright:      JarvisLee
    Date:           2022/01/20
    Filename:       Convertor.py
    Description:    Implement a convertor to transform the parameters of vision-outlooker to match the vision-encoder.
'''

# Import the necessary library.
import sys
sys.path.append('./')
import torch
from collections import OrderedDict
from VOLOParams.Original.VOLO.models import volo_d1, volo_d2, volo_d3, volo_d4, volo_d5
from Utils.ParamsHandler import Handler

# Get the hyperparameters.
Cfg = Handler.Parser(Handler.Generator(paramsDir = './Params.txt'))

# Create the convertion dictionary.
convertDict = {
    'pos_embed':'posEmbed', 'patch_embed':'patchEmbed', 'conv':'patchEmbed', 
    'proj':'projLayer', 'network':'volo', 'norm1':'layerNorm1', 
    'norm2':'layerNorm2', 'fc1':'projLayer1', 'fc2':'projLayer2', 
    'v':'valueLayer', 'attn':None
}

# Indicate whether the Vision-Outlooker state is valid or not.
assert Cfg.volo >= 1 and Cfg.volo <= 5, 'Please set a valid Vision-Outlooker state! (1 <= volo <= 5)'
# Set the hyperparameters for the captioning and answering with transformer architecture according to the Vision-Outlooker state.
if Cfg.volo == 1:
    stageSizes = [4, 4, 8, 2, 8]
    headSizes = [6, 12, 12, 12, 8]
    embedSizes = [192, 384, 384, 384, 384]
    mlpRatios = [3, 3, 3, 3, 3]
    model = volo_d1()
    originalName = f'{Cfg.voloDir}//Original//d1_224_84.2.pth.tar'
    convertName = f'{Cfg.voloDir}//VOLO-Type-1-224.pt'
elif Cfg.volo == 2:
    stageSizes = [6, 4, 10, 4, 8]
    headSizes = [8, 16, 16, 16, 8]
    embedSizes = [256, 512, 512, 512, 512]
    mlpRatios = [3, 3, 3, 3, 3]
    model = volo_d2()
    originalName = f'{Cfg.voloDir}//Original//d2_224_85.2.pth.tar'
    convertName = f'{Cfg.voloDir}//VOLO-Type-2-224.pt'
elif Cfg.volo == 3:
    stageSizes = [8, 8, 16, 4, 12]
    headSizes = [8, 16, 16, 16, 12]
    embedSizes = [256, 512, 512, 512, 512]
    mlpRatios = [3, 3, 3, 3, 3]
    model = volo_d3()
    originalName = f'{Cfg.voloDir}//Original//d3_224_85.4.pth.tar'
    convertName = f'{Cfg.voloDir}//VOLO-Type-3-224.pt'
elif Cfg.volo == 4:
    stageSizes = [8, 8, 16, 4, 12]
    headSizes = [12, 16, 16, 16, 12]
    embedSizes = [384, 768, 768, 768, 768]
    mlpRatios = [3, 3, 3, 3, 3]
    model = volo_d4()
    originalName = f'{Cfg.voloDir}//Original//d4_224_85.7.pth.tar'
    convertName = f'{Cfg.voloDir}//VOLO-Type-4-224.pt'
else:
    stageSizes = [12, 12, 20, 4, 16]
    headSizes = [12, 16, 16, 16, 16]
    embedSizes = [384, 768, 768, 768, 768]
    mlpRatios = [4, 4, 4, 4, 4]
    model = volo_d5()
    originalName = f'{Cfg.voloDir}//Original//d5_224_86.10.pth.tar'
    convertName = f'{Cfg.voloDir}//VOLO-Type-5-224.pt'

# Convert the parameters of the vision-outlooker.
class Convertor():
    '''
        Convert the parameters of the vision-outlooker.\n
        Contains one part.\n
            - 'Convert' is used to convert the parameters of the vision-outlooker.
    '''

    # Convert the parameters of the vision-outlooker.
    def Convert(key, convertDict):
        '''
            Convert the parameters of the vision-outlooker.\n
            Params:\n
                - 'key' is the key of the parameters in the original vision-outlooker.
                - 'convertDict' is the dictionary to convert the key to be the target.
        '''
        # Set the target.
        target = []
        # Convert the key to be the target.
        for i, each in enumerate(key):
            # Check whether the part of key is set in the convert dictionary or not.
            if each in convertDict.keys():
                # Convert the parameters.
                if each == 'attn':
                    if i == (len(key) - 2):
                        target.append('attenScore')
                    else:
                        target.append('atten')
                else:
                    target.append(convertDict[each])
            else:
                if each == 'cls_token' or each == 'post_network' or each == 'aux_head' or each == 'norm' or each == 'head':
                    return None
                target.append(each)
        target = '.'.join(target)
        # Return the target.
        return target

# Convert the model parameters.
if __name__ == "__main__":
    # Initialize the target parameters.
    targetParams = OrderedDict()
    # Load the original parameters.
    model.load_state_dict(torch.load(originalName, map_location = 'cpu'))
    print(model)
    input("PAUSE")
    # Convert the parameters.
    for key in model.state_dict():
        # Check whether the key is the downsampling layer or not.
        if key == 'network.1.proj.weight':
            targetParams['volo.1.downSample.weight'] = model.state_dict()[key]
        elif key == 'network.1.proj.bias':
            targetParams['volo.1.downSample.bias'] = model.state_dict()[key]
        else:
            # Split the key into the list.
            tempKey = key.split('.')
            # Convert the key into the target key.
            targetKey = Convertor.Convert(key = tempKey, convertDict = convertDict)
            # Save the converted parameters.
            if targetKey == None:
                continue
            else:
                targetParams[targetKey] = model.state_dict()[key]
    # Save the converted parameters.
    torch.save(targetParams, convertName)