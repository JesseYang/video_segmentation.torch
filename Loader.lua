require 'nn'
require 'torch'
require 'lmdb'
require 'xlua'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')

local indexer = torch.class('indexer')

function indexer:__init(dirPath, batchSize)
    local dbClip = lmdb.env { Path = dirPath .. '/clip', Name = 'clip' }
    local dbLabel = lmdb.env { Path = dirPath .. '/label', Name = 'label' }

    self.batchSize = batchSize
    self.count = 1
    -- get the size of lmdb
    dbClip:open()
    dbLabel:open()
    local clipLMDBSize = dbClip:stat()['entries']
    local labelLMDBSize = dbLabel:stat()['entries']
    self.size = clipLMDBSize
    dbClip:close()
    dbLabel:close()
    self.nbOfBatches = math.floor(self.size / self.batchSize)
    assert(clipLMDBSize == labelLMDBSize, 'Audio and transcript LMDBs had different lengths!')
    assert(self.size > self.batchSize, 'batchSize larger than lmdb size!')

    print("Number of samples: " .. self.size)
    print("Number of samples in each epoch: " .. self.nbOfBatches * self.batchSize)

    self.inds = torch.range(1, self.size):split(batchSize)
    self.batchIndices = torch.range(1, self.nbOfBatches)
end

function indexer:nextIndices()
    -- if self.count > #self.inds then self.count = 1 end
    if self.count > self.nbOfBatches then self.count = 1 end
    local index = self.batchIndices[self.count]
    local inds = self.inds[index]
    self.count = self.count + 1
    return inds
end

function indexer:permuteIndices()
    self.inds = torch.randperm(self.size):split(self.batchSize)
end

function indexer:permuteBatchOrder()
    self.batchIndices = torch.randperm(self.nbOfBatches)
end

local Loader = torch.class('Loader')

function Loader:__init(dirPath)
    self.dbClip = lmdb.env { Path = dirPath .. '/clip', Name = 'clip' }
    self.dbLabel = lmdb.env { Path = dirPath .. '/label', Name = 'label' }
    self.dbClip:open()
    self.size = self.dbClip:stat()['entries']
    self.dbClip:close()
end

function Loader:nextBatch(indices)
    local tensors = {}
    local labels = {}

    local maxLength = 0
    local freq = 0

    self.dbClip:open(); local readerClip = self.dbClip:txn(true) -- readonly
    self.dbLabel:open(); local readerLabel = self.dbLabel:txn(true)

    local size = indices:size(1)

    local permutedIndices = torch.randperm(size) -- batch tensor has different order each time
    -- reads out a batch and store in lists
    local channel = 0
    local height = 0
    local width = 0
    for x = 1, size do
        -- local ind = indices[permutedIndices[x]]
        local ind = indices[x]
        local input = readerClip:get(ind):float()
        local label = readerLabel:get(ind)

        channel = input:size(1)
        height = input:size(2)
        width = input:size(3)

        table.insert(tensors, input)
        table.insert(labels, label)
    end

    local inputs = torch.Tensor(size, channel, height, width):zero()
    local targets = torch.Tensor(size):zero()
    for ind, tensor in ipairs(tensors) do
        inputs[ind]:copy(tensor)
    end
    for ind, label in ipairs(labels) do
        targets[ind] = label
    end

    readerClip:abort(); self.dbClip:close()
    readerLabel:abort(); self.dbLabel:close()

    return inputs, targets
end
