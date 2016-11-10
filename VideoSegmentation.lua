require 'UtilsMultiGPU'

-- Creates the covnet+rnn structure.
local function videoSegmentation(opt)
    local conv = nn.Sequential()

    -- insert the input channel size: 1
    table.insert(opt.channels, 1, 1)
    for x = 1, #opt.kernel_heights do
        -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
        conv:add(nn.SpatialConvolution(opt.cnn.channels[x],
                                       opt.cnn.channels[x + 1],
                                       opt.cnn.kernel_widths[x],
                                       opt.cnn.kernel_heights[x],
                                       1,
                                       1))
        if opt.with_bn then
            conv:add(nn.SpatialBatchNormalization(opt.channels[x + 1]))
        end
        conv:add(nn.Clamp(0, 20))
    end

    local fullyConnected = nn.Sequential()
    fullyConnected:add(nn.BatchNormalization(rnnHiddenSize))
    -- fullyConnected:add(nn.Linear(rnnHiddenSize, opt.label_size))

    local model = nn.Sequential()
    model:add(conv)
    model:add(nn.Bottle(fullyConnected, 2))
    model = makeDataParallel(model, opt.nGPU)
    return model
end

return videoSegmentation