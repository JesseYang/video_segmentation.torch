-- Expects data in the format of <root><train/test><datasetname><filename.wav/filename.txt>
-- Creates an LMDB of everything in these folders into a train and test set.

require 'lfs'
require 'xlua'
require 'lmdb'
require 'torch'
require 'parallel'
require 'image'

local tds = require 'tds'

local cmd = torch.CmdLine()
cmd:option('-rootPath', 'prepare_datasets/dataset', 'Path to the data')
cmd:option('-lmdbPath', 'prepare_datasets/lmdb', 'Path to save LMDBs to')
cmd:option('-frameNum', 11, 'Number of continuous frames for one sample')
cmd:option('-skip', 3, 'Number of frames to skip in the beginning of the video')
cmd:option('-videoExtension', 'avi', 'The extension of the video files (avi/mp4)')
cmd:option('-processes', 3, 'Number of processes used to create LMDB')

local opt = cmd:parse(arg)
local dataPath = opt.rootPath
local lmdbPath = opt.lmdbPath
local extension = '.' .. opt.videoExtension
parallel.nfork(opt.processes)

local function startWriter(path, name)
    local db = lmdb.env {
        Path = path,
        Name = name
    }
    db:open()
    local txn = db:txn()
    return db, txn
end

local function closeWriter(db, txn)
    txn:commit()
    db:close()
end

local function createLMDB(dataPath, lmdbPath, id)
    local vecs = tds.Vec()

    local size = tonumber(sys.execute("find " .. dataPath .. " -type f -name '*'" .. extension .. " | wc -l "))
    vecs:resize(size)


    local files = io.popen("find -L " .. dataPath .. " -type f -name '*" .. extension .. "'")
    local counter = 1
    local buffer = tds.Vec()
    buffer:resize(size)

    for file in files:lines() do
        buffer[counter] = file
        counter = counter + 1
    end


    local function getSize(opts)
        local audioFilePath = opts.file
        local transcriptFilePath = opts.file:gsub(opts.extension, ".txt")
        local opt = opts.opt
        return { audioFilePath, transcriptFilePath }
    end

    for x = 1, opt.processes do
        local opts = { extension = extension, file = buffer[x], opt = opt }
        parallel.children[x]:send({ opts, getSize })
    end

    local processCounter = 1
    for x = 1, size do
        local result = parallel.children[processCounter]:receive()
        vecs[x] = tds.Vec(unpack(result))
        -- xlua.progress(x, size)
        if x % 1000 == 0 then collectgarbage() end
        -- send next index to retrieve
        if x + opt.processes <= size then
            local opts = { extension = extension, file = buffer[x + opt.processes], opt = opt }
            parallel.children[processCounter]:send({ opts, getSize })
        end
        if processCounter == opt.processes then
            processCounter = 1
        else
            processCounter = processCounter + 1
        end
    end

    local size = #vecs

    print("Creating LMDB dataset to: " .. lmdbPath)
    -- start writing
    local dbClip, readerClip = startWriter(lmdbPath .. '/clip', 'clip')
    local dbLabel, readerLabel = startWriter(lmdbPath .. '/label', 'label')


    local function getData(opts)
        local videoFilePath = opts.file
        local labelFilePath = opts.file:gsub(opts.extension, ".txt")
        local status, height, width, length, fps = opts.video.init(videoFilePath)

        local label_content
        for line in io.lines(labelFilePath) do
            label_content = line
        end

        -- read data for the first sample
        clips = { }
        labels = { }
        start_idx = 1 + opts.opt.skip + (opts.opt.frameNum - 1) / 2
        clip = torch.ByteTensor(opts.opt.frameNum, height, width)
        frame = torch.ByteTensor(3, height, width)
        -- skip the first opts.opt.skip frames
        for x = 1, opts.opt.skip do
            status = opts.video.frame_rgb(frame)
            if status == false then
                opts.video.exit()
                return { clips, labels }
            end
        end
        for x = 1, opts.opt.frameNum do
            status = opts.video.frame_rgb(frame)
            if status == false then
                opts.video.exit()
                return { clips, labels }
            end
            clip[x] = image.rgb2y(frame)
        end
        target_idx = 1 + opts.opt.skip + (opts.opt.frameNum - 1) / 2
        new_clip = clip:clone()
        clips[#clips + 1] = new_clip - new_clip:mean()
        labels[#labels + 1] = torch.Tensor(1):fill(tonumber(label_content:sub(target_idx, target_idx)))

        -- iteratively read next frame data until the end of the video
        while true do
            target_idx = target_idx + 1
            status = opts.video.frame_rgb(frame)
            if status == false then
                opts.video.exit()
                return { clips, labels }
            end
            gray_frame = image.rgb2y(frame)
            for x = 1, opts.opt.frameNum - 1 do
                clip[x] = clip[x + 1]
            end
            clip[opts.opt.frameNum] = gray_frame
            new_clip = clip:clone()
            clips[#clips + 1] = new_clip - new_clip:mean()
            labels[#labels + 1] = torch.Tensor(1):fill(tonumber(label_content:sub(target_idx, target_idx)))
        end

        return { clips, labels }
    end

    for x = 1, opt.processes do
        local opts = { extension = extension, file = buffer[x], opt = opt }
        parallel.children[x]:send({ opts, getData })
    end

    local processCounter = 1
    local idx = 1
    for x = 1, size do
        local result = parallel.children[processCounter]:receive()
        local clips, labels = unpack(result)

        for i = 1, #clips do
            readerClip:put(idx, clips[i])
            readerLabel:put(idx, labels[i])
            idx = idx + 1
        end

        -- if x % 500 == 0 then
        readerClip:commit(); readerClip = dbClip:txn()
        readerLabel:commit(); readerLabel = dbLabel:txn()
        collectgarbage()
        -- end

        if x + opt.processes <= size then
            local opts = { extension = extension, file = buffer[x + opt.processes], opt = opt }
            parallel.children[processCounter]:send({ opts, getData })
        end
        if processCounter == opt.processes then
            processCounter = 1
        else
            processCounter = processCounter + 1
        end
        xlua.progress(x, size)
    end

    closeWriter(dbClip, readerClip)
    closeWriter(dbLabel, readerLabel)
end

function parent()
    local function looper()
        require 'torch'
        require 'image'
        local video = assert(require("libvideo_decoder"))
        while true do
            local object = parallel.parent:receive()
            local opts, code = unpack(object)
            opts.video = video
            local result = code(opts)
            parallel.parent:send(result)
            collectgarbage()
        end
    end

    parallel.children:exec(looper)

    createLMDB(dataPath .. '/train', lmdbPath .. '/train', 'train')
    createLMDB(dataPath .. '/test', lmdbPath .. '/test', 'test')
    parallel.close()
end

local ok, err = pcall(parent)
if not ok then
    print(err)
    parallel.close()
end
