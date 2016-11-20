require 'nn'
require 'UtilsMultiGPU'
local json = require 'json'
video = assert(require("libvideo_decoder"))
local cmd = torch.CmdLine()
cmd:option('-modelPath', 'video_segmentation.t7', 'Path of model to load')
cmd:option('-videoPath', '', 'Path to the input image to predict on')
cmd:option('-frameNum', 11, 'Number of continuous frames for one sample')
cmd:option('-nGPU', 1)

local opt = cmd:parse(arg)

if opt.nGPU > 0 then
    require 'cunn'
    require 'cudnn'
end

local model =  loadDataParallel(opt.modelPath, opt.nGPU)

local net_opt = json.load('params.json')

local status, height, width, length, fps = video.init(opt.videoPath)

assert(height == net_opt.input_height, 'video height must be equal to the Network input_height')
assert(width == net_opt.input_width, 'video width must be equal to the Network input_width')
assert(opt.frameNum == net_opt.input_channel, 'frameNum must be equal to the Network input_channel')

local result = { }
local start_frame_idx = 1 + (opt.frameNum - 1) / 2
for x = 1, start_frame_idx - 1 do
    table.insert(result, '-')
end


local clip = torch.ByteTensor(opt.frameNum, height, width)
local frame_idx = 0
while true do
    frame_idx = frame_idx + 1
    status = video.frame_rgb(frame)
    if status == false then
        video.exit()
        break
    end
    if frame_idx < opt.frameNum then
        clip[frame_idx + 1] = image.rgb2y(frame)
        goto continue
    end

    for x = 1, opt.frameNum - 1 do
        clip[x] = clip[x + 1]
    end
    clip[opt.frameNum] = image.rgb2y(frame)

    -- local target_frame_idx = frame_idx - (opt.frameNum - 1) / 2
    local predictions = model:forward(clip)
    -- parse the prediction and insert to result table

    ::continue::
end

-- if opt.nGPU > 0 then
--     img = img:cuda()
--     model = model:cuda()
-- end

-- model:evaluate()
-- local predictions = model:forward(img)
