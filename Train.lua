local Network = require 'Network'
local json = require 'json'

-- Options can be overrided on command line run.
local cmd = torch.CmdLine()
cmd:option('-loadModel', true, 'Load previously saved model')
cmd:option('-saveModel', true, 'Save model after training/testing')
cmd:option('-modelName', 'VideoSegmentation', 'Name of class containing architecture')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-trainingSetLMDBPath', './prepare_datasets/lmdb/train/', 'Path to LMDB training dataset')
cmd:option('-validationSetLMDBPath', './prepare_datasets/lmdb/test/', 'Path to LMDB test dataset')
cmd:option('-logsTrainPath', './logs/TrainingLoss/', ' Path to save Training logs')
cmd:option('-logsValidationPath', './logs/ValidationScores/', ' Path to save Validation logs')
cmd:option('-saveModelInTraining', false, 'save model periodically through training')
cmd:option('-modelTrainingPath', './models/', ' Path to save periodic training models')
cmd:option('-saveModelIterations', 50, 'When to save model through training')
cmd:option('-modelPath', 'video_segmentation.t7', 'Path of final model to save/load')
-- cmd:option('-dictionaryPath', './equ_dictionary', ' File containing the dictionary to use')
cmd:option('-epochs', 10, 'Number of epochs for training')
cmd:option('-learningRate', 1e-4, ' Training learning rate')
cmd:option('-learningRateAnnealing', 1.1, 'Factor to anneal lr every epoch')
-- cmd:option('-maxNorm', 400, 'Max norm used to normalize gradients')
cmd:option('-momentum', 0.90, 'Momentum for SGD')
cmd:option('-batchSize', 10, 'Batch size in training')
cmd:option('-validationBatchSize', 10, 'Batch size for validation')

local opt = cmd:parse(arg)

local net_opt = json.load('params.json')

for k,v in pairs(net_opt) do opt[k] = v end

--Parameters for the stochastic gradient descent (using the optim library).
local optimParams = {
    learningRate = opt.learningRate,
    learningRateAnnealing = opt.learningRateAnnealing,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}

--Create and train the network based on the parameters and training data.
Network:init(opt)

Network:trainNetwork(opt.epochs, optimParams, opt)

--Creates the loss plot.
Network:createLossGraph()
