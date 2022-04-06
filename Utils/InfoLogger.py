'''
    Copyright:      JarvisLee
    Date:           2022/01/19
    Filename:       InfoLogger.py
    Description:    Log the information when training the model.
'''

# Import the necessary library.
import logging
from visdom import Visdom

# Encapsulate the information logger's tools.
class Logger():
    '''
        Encapsulate all the functions which are used to log the training details.\n
        Contains five parts:\n
            - 'VisConfigurator' configurates the visdom server.
            - 'LogConfigurator' configurates the information logger.
            - 'VisDrawer' draws the graphs in the visdom server.
            - 'VisSaver' saves the visdom graphs.
    '''
    
    # Configurates the visdom.
    def VisConfigurator(currentTime = None, visName = 'GraphLogging'):
        '''
            Configurates the visdom.\n
            Params:\n
                - 'currentTime' indicates each training graph.
                - 'visName' sets the name of the visdom environment.
        '''
        # Create the new visdom environment.
        vis = Visdom(env = visName)
        # Initialize the graphs.
        lossGraph = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainLoss', 'EvalLoss'], xlabel = 'Epoch', ylabel = 'Loss', title = f'Train and Eval Losses - {currentTime}'), name = 'TrainLoss')
        accGraph = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainAcc', 'EvalAcc'], xlabel = 'Epoch', ylabel = 'Acc', title = f'Train and Eval Accs - {currentTime}'), name = 'TrainAcc')
        bleuGraphv1 = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainBLEUv1', 'EvalBLEUv1'], xlabel = 'Epoch', ylabel = 'BLEU-1', title = f'Train and Eval BLEUv1 - {currentTime}'), name = 'TrainBLEUv1')
        bleuGraphv2 = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainBLEUv2', 'EvalBLEUv2'], xlabel = 'Epoch', ylabel = 'BLEU-2', title = f'Train and Eval BLEUv2 - {currentTime}'), name = 'TrainBLEUv2')
        bleuGraphv3 = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainBLEUv3', 'EvalBLEUv3'], xlabel = 'Epoch', ylabel = 'BLEU-3', title = f'Train and Eval BLEUv3 - {currentTime}'), name = 'TrainBLEUv3')
        bleuGraphv4 = vis.line(Y = [0], X = [1], opts = dict(legend = ['TrainBLEUv4', 'EvalBLEUv4'], xlabel = 'Epoch', ylabel = 'BLEU-4', title = f'Train and Eval BLEUv4 - {currentTime}'), name = 'TrainBLEUv4')
        vis.line(Y = [0], X = [1], win = lossGraph, update = 'append', name = 'EvalLoss')
        vis.line(Y = [0], X = [1], win = accGraph, update = 'append', name = 'EvalAcc')
        vis.line(Y = [0], X = [1], win = bleuGraphv1, update = 'append', name = 'EvalBLEUv1')
        vis.line(Y = [0], X = [1], win = bleuGraphv2, update = 'append', name = 'EvalBLEUv2')
        vis.line(Y = [0], X = [1], win = bleuGraphv3, update = 'append', name = 'EvalBLEUv3')
        vis.line(Y = [0], X = [1], win = bleuGraphv4, update = 'append', name = 'EvalBLEUv4')
        # Return the visdom.
        return vis, lossGraph, accGraph, bleuGraphv1, bleuGraphv2, bleuGraphv3, bleuGraphv4
    
    # Configurates the logger.
    def LogConfigurator(logDir, filename, format = "%(asctime)s %(levelname)s %(message)s", dateFormat = "%Y-%m-%d %H:%M:%S %p"):
        '''
            Configurates the logger.\n
            Params:\n
                - 'logDir' is the directory of the logging file.
                - 'filename' is the whole name of the logging file.
                - 'format' is the formate of the logging info.
                - 'dateFormate' is the formate of the data info.
        '''
        # Create the logger.
        logger = logging.getLogger()
        # Set the level of the logger.
        logger.setLevel(logging.INFO)
        # Set the logging file.
        file = logging.FileHandler(filename = logDir + '/' + filename, mode = 'a')
        # Set the level of the logging file.
        file.setLevel(logging.INFO)
        # Set the logging console.
        console = logging.StreamHandler()
        # Set the level of the logging console.
        console.setLevel(logging.WARNING)
        # Set the logging format.
        fmt = logging.Formatter(fmt = format, datefmt = dateFormat)
        file.setFormatter(fmt)
        console.setFormatter(fmt)
        # Add the logging file into the logger.
        logger.addHandler(file)
        logger.addHandler(console)
        # Return the logger.
        return logger
    
    # Draw the graphs.
    def VisDrawer(vis, epoch, trainLoss, evalLoss, trainAcc, evalAcc, trainBLEUv1, evalBLEUv1, trainBLEUv2, evalBLEUv2, trainBLEUv3, evalBLEUv3, trainBLEUv4, evalBLEUv4):
        '''
            Draw the graph in visdom.\n
            Params:\n
                - 'vis' is the tuple contains visdom, lossGraph and accGraph.
                - 'epoch' is the current training epoch.
                - 'trainLoss' is the training loss.
                - 'evalLoss' is the evaluating loss.
                - 'trainAcc' is the training accuracy.
                - 'evalAcc' is the evaluating accuracy.
                - 'trainBLEUvX' is the xth training BLEU score.
                - 'evalBLEUvX' is the xth evaluating BLEU score.
        '''
        # Inidicate whether the parameters are valid.
        assert type(vis[0]) is type(Visdom()), 'The vis must be the visdom environment!'
        assert type(epoch) is int, 'The epoch must be an integer!'
        # Draw the graph.
        if epoch == 1:
            vis[0].line(Y = [trainLoss], X = [epoch], win = vis[1], name = 'TrainLoss', update = 'new')
            if evalLoss != None:
                vis[0].line(Y = [evalLoss], X = [epoch], win = vis[1], name = 'EvalLoss', update = 'new')
            vis[0].line(Y = [trainAcc], X = [epoch], win = vis[2], name = 'TrainAcc', update = 'new')
            if evalAcc != None:
                vis[0].line(Y = [evalAcc], X = [epoch], win = vis[2], name = 'EvalAcc', update = 'new')
            vis[0].line(Y = [trainBLEUv1], X = [epoch], win = vis[3], name = 'TrainBLEUv1', update = 'new')
            vis[0].line(Y = [trainBLEUv2], X = [epoch], win = vis[4], name = 'TrainBLEUv2', update = 'new')
            vis[0].line(Y = [trainBLEUv3], X = [epoch], win = vis[5], name = 'TrainBLEUv3', update = 'new')
            vis[0].line(Y = [trainBLEUv4], X = [epoch], win = vis[6], name = 'TrainBLEUv4', update = 'new')
            if evalBLEUv1 != None:
                vis[0].line(Y = [evalBLEUv1], X = [epoch], win = vis[3], name = 'EvalBLEUv1', update = 'new')
            if evalBLEUv2 != None:
                vis[0].line(Y = [evalBLEUv2], X = [epoch], win = vis[4], name = 'EvalBLEUv2', update = 'new')
            if evalBLEUv3 != None:
                vis[0].line(Y = [evalBLEUv3], X = [epoch], win = vis[5], name = 'EvalBLEUv3', update = 'new')
            if evalBLEUv4 != None:
                vis[0].line(Y = [evalBLEUv4], X = [epoch], win = vis[6], name = 'EvalBLEUv4', update = 'new')
        else:
            vis[0].line(Y = [trainLoss], X = [epoch], win = vis[1], name = 'TrainLoss', update = 'append')
            if evalLoss != None:
                vis[0].line(Y = [evalLoss], X = [epoch], win = vis[1], name = 'EvalLoss', update = 'append')
            vis[0].line(Y = [trainAcc], X = [epoch], win = vis[2], name = 'TrainAcc', update = 'append')
            if evalAcc != None:
                vis[0].line(Y = [evalAcc], X = [epoch], win = vis[2], name = 'EvalAcc', update = 'append')
            vis[0].line(Y = [trainBLEUv1], X = [epoch], win = vis[3], name = 'TrainBLEUv1', update = 'append')
            vis[0].line(Y = [trainBLEUv2], X = [epoch], win = vis[4], name = 'TrainBLEUv2', update = 'append')
            vis[0].line(Y = [trainBLEUv3], X = [epoch], win = vis[5], name = 'TrainBLEUv3', update = 'append')
            vis[0].line(Y = [trainBLEUv4], X = [epoch], win = vis[6], name = 'TrainBLEUv4', update = 'append')
            if evalBLEUv1 != None:
                vis[0].line(Y = [evalBLEUv1], X = [epoch], win = vis[3], name = 'EvalBLEUv1', update = 'append')
            if evalBLEUv2 != None:
                vis[0].line(Y = [evalBLEUv2], X = [epoch], win = vis[4], name = 'EvalBLEUv2', update = 'append')
            if evalBLEUv3 != None:
                vis[0].line(Y = [evalBLEUv3], X = [epoch], win = vis[5], name = 'EvalBLEUv3', update = 'append')
            if evalBLEUv4 != None:
                vis[0].line(Y = [evalBLEUv4], X = [epoch], win = vis[6], name = 'EvalBLEUv4', update = 'append')
    
    # Close the visdom server.
    def VisSaver(vis, visName = 'GraphLogging'):
        '''
            Close the visdom server.\n
            Params:\n
                - 'visName' sets the name of the visdom environment.
        '''
        # Indicate whether the parameters are valid.
        assert type(vis[0]) is type(Visdom()), 'The vis must be the visdom environment!'
        # Save the graph.
        vis[0].save(envs = [visName])