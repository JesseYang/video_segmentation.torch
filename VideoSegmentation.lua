require 'UtilsMultiGPU'

-- Creates the covnet+rnn structure.
local function videoSegmentation(opt)
    local conv = nn.Sequential()

    -- insert the input channel size: opt.input_channel
    table.insert(opt.channels, 1, opt.input_channel)
    local pad_width = 0
    local pad_height = 0
    for x = 1, #opt.kernel_heights do
        -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH], [dilationW], [dilationH]) conv layers.
        conv:add(nn.SpatialDilatedConvolution(opt.channels[x],
                                              opt.channels[x + 1],
                                              opt.kernel_widths[x],
                                              opt.kernel_heights[x],
                                              1,
                                              1,
                                              0,
                                              0,
                                              opt.dilations[x],
                                              opt.dilations[x]))
        pad_width = pad_width + (opt.kernel_widths[x] - 1) * opt.dilations[x]
        pad_height = pad_height + (opt.kernel_heights[x] - 1) * opt.dilations[x]
        if opt.with_bn then
            conv:add(nn.SpatialBatchNormalization(opt.channels[x + 1]))
        end
        conv:add(nn.Clamp(0, 20))
    end

    local fullyConnected = nn.Sequential()
    local cnn_out_channel = opt.channels[#opt.kernel_heights + 1]
    local cnn_out_size = (opt.input_width - pad_width) * (opt.input_height - pad_height)
    local cnn_out_unit = cnn_out_size * cnn_out_channel
    fullyConnected:add(nn.Linear(cnn_out_unit, opt.label_size))

    local model = nn.Sequential()
    model:add(conv)
    -- model:add(nn.View(opt.batchSize, -1))
    model:add(nn.View(-1, cnn_out_unit))
    model:add(fullyConnected, 2)
    model = makeDataParallel(model, opt.nGPU)
    return model
end

return videoSegmentation