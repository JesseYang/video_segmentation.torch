require 'Loader'
require 'torch'
require 'xlua'
local threads = require 'threads'
-- require 'SequenceError'

local ModelEvaluator = torch.class('ModelEvaluator')

local loader

function ModelEvaluator:__init(isGPU, datasetPath, testBatchSize, logsPath)
    loader = Loader(datasetPath)
    self.testBatchSize = testBatchSize
    self.nbOfTestIterations = math.floor(loader.size / testBatchSize)
    self.indexer = indexer(datasetPath, testBatchSize)
    self.pool = threads.Threads(1, function() require 'Loader' end)
    self.logsPath = logsPath
    self.suffix = '_' .. os.date('%Y%m%d_%H%M%S')
    -- self.sequenceError = SequenceError()
    self.input = torch.Tensor()
    self.isGPU = isGPU
    if isGPU then
        self.input = self.input:cuda()
    end
end

function ModelEvaluator:runEvaluation(model, verbose, epoch)
    local clipBuf, labelBuf

    -- get first batch
    local inds = self.indexer:nextIndices()
    self.pool:addjob(function()
        return loader:nextBatch(inds)
    end,
        function(clip, label)
            clipBuf = clip
            labelBuf = label
        end)

    if verbose then
        local f = assert(io.open(self.logsPath .. 'WER_Test' .. self.suffix .. '.log', 'a'), "Could not create validation test logs, does the folder "
                .. self.logsPath .. " exist?")
        f:write('======================== BEGIN WER TEST EPOCH: ' .. epoch .. ' =========================\n')
        f:close()
    end

    local evaluationPredictions = {} -- stores the predictions to order for log.
    local cumER = 0
    local numberOfSamples = 0
    -- ======================= for every test iteration ==========================
    for i = 1, self.nbOfTestIterations do
        -- get buf and fetch next one
        self.pool:synchronize()
        local inputsCPU, targets = clipBuf, labelBuf
        inds = self.indexer:nextIndices()
        self.pool:addjob(function()
            return loader:nextBatch(inds)
        end,
            function(clip, label)
                clipBuf = clip
                labelBuf = label
            end)

        self.input:resize(inputsCPU:size()):copy(inputsCPU)
        local predictions = model:forward(self.input)
        if self.isGPU then cutorch.synchronize() end

        local size = predictions:size(1)
        for j = 1, size do
            local _, prediction = torch.max(predictions[j], 1)
            local target = labelBuf[j]

            if prediction[1] ~= target then
                cumER = cumER + 1
            end
        end
        numberOfSamples = numberOfSamples + size
    end

    local averageER = cumER / numberOfSamples

    local f = assert(io.open(self.logsPath .. 'Evaluation_Test' .. self.suffix .. '.log', 'a'))
    f:write(string.format("Average ER = %.2f", averageER * 100))
    f:close()

    self.pool:synchronize() -- end the last loading
    return averageER
end
