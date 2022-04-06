'''
    Copyright:      JarvisLee
    Date:           2022/01/15
    Filename:       ParamsHandler.py
    Description:    Handle the hyperparameters.
'''

# Import the necessary library.
import argparse
from easydict import EasyDict as Config

# Encapsulate the handler's tools.
class Handler():
    '''
        Encapsulate all the functions which are used to handle the hyperparameters.\n
        Contains four parts:\n
            - 'Convertor' converts the type of each hyperparameter.
            - 'Generator' generates the initial hyperparameters from the Params.txt file.
            - 'Parser' parses the hyperparameters.
            - 'Displayer' displays the hyperparameters.
    '''

    # Convert the type of the hyperparameter.
    def Convertor(param):
        '''
            Convert the type of the hyperparameters.\n
            Params:\n
                - 'param' is the hyperparameters.
        '''
        # Convert the hyperparameters.
        try:
            param = eval(param)
        except:
            param = param
        # Return the hyperparameters.
        return param

    # Generate the configurator of the hyperparameters.
    def Generator(paramsDir = './Params.txt'):
        '''
            Generate the configurator of the hyperparameters.\n
            Params:\n
                - 'paramDir' is the directory of the hyperparameters' default setting file.
        ''' 
        # Create the configurator of the hyperparameters.
        Cfg = Config()
        # Get the names of the hyperparameters.
        with open(paramsDir) as file:
            lines = file.readlines()
            # Initialize the hyperparameters.
            for line in lines:
                Cfg[line.split("\n")[0].split("->")[0]] = Handler.Convertor(line.split("\n")[0].split("->")[1])
        # Return the dictionary of the hyperparameters.
        return Cfg
    
    # Parse the hyperparameters.
    def Parser(Cfg):
        '''
            Parse the hyperparameters.\n
            Params:\n
                - 'Cfg' is the configurator of the hyperparameters.
        '''
        # Indicate whether the Cfg is a configurator or not.
        assert type(Cfg) is type(Config()), 'Please input the configurator!'
        # Create the hyperparameters' parser.
        parser = argparse.ArgumentParser(description = 'Hyperparameters Parser')
        # Add the hyperparameters into the parser.
        for param in Cfg.keys():
            parser.add_argument(f'-{param}', f'--{param}', f'-{param.lower()}', f'--{param.lower()}', f'-{param.upper()}', f'--{param.upper()}', dest = param, type = type(Cfg[param]), default = Cfg[param], help = f'The type of {param} is {type(Cfg[param])}')
        # Parse the hyperparameters.
        params = vars(parser.parse_args())
        # Update the configurator.
        Cfg.update(params)
        # Return the configurator.
        return Cfg
    
    # Display the hyperparameters setting.
    def Displayer(Cfg):
        '''
            Display the hyperparameters.\n
            Params:\n
                - 'Cfg' is the configurator.
        '''
        # Indicate whether the Cfg is a configurator or not.
        assert type(Cfg) is type(Config()), 'Please input the configurator!'
        # Set the displayer.
        displayer = [''.ljust(20) + f'{param}:'.ljust(30) + f'{Cfg[param]}' for param in Cfg.keys()]
        # Return the results of the displayer.
        return "\n".join(displayer)