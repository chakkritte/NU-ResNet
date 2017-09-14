--
--  Copyright (c) 2017, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNeXt model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = cudnn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local DepthConvolution = nn.SpatialDepthWiseConvolution

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

local function conv_bn(inp, oup, stride)
	local conv = nn.Sequential()
	conv:add(Convolution(inp, oup, 3,3 ,stride,stride, 1,1))
	conv:add(SBatchNorm(oup))
	conv:add(ReLU(true))
	return conv

end

local function conv_dw(inp, oup, stride)
	local conv = nn.Sequential()
	--conv:add(Convolution(inp, inp, 3,3, stride, stride, 1 , 1 , inp))
	conv:add(DepthConvolution(inp, 1, 3,3, stride, stride, 1 , 1))
	conv:add(SBatchNorm(inp))
	conv:add(ReLU(true))

	conv:add(Convolution(inp, oup, 1 ,1 , 1 , 1, 0 , 0))
	conv:add(SBatchNorm(oup))
	conv:add(ReLU(true))

	return conv
end

   
   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')
	  
	  
      model = nn.Sequential()
      model:add(conv_bn(3,  32, 2))
      model:add(conv_dw(32,  64, 1))
      model:add(conv_dw(64,  128, 2))
      model:add(conv_dw(128, 128, 1))
      model:add(conv_dw(128, 256, 2))
      model:add(conv_dw(256, 256, 1))
      model:add(conv_dw(256, 512, 2))
      model:add(conv_dw(512, 512, 1))
      model:add(conv_dw(512, 512, 1))
      model:add(conv_dw(512, 512, 1))
      model:add(conv_dw(512, 512, 1))
      model:add(conv_dw(512, 512, 1))
      model:add(conv_dw(512, 1024, 2))
      model:add(conv_dw(1024, 1024, 1))
      model:add(Avg(7, 7, 1, 1))

      model:add(nn.View(1024):setNumInputDims(3))
      model:add(nn.Linear(1024, opt.nClasses))

   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel