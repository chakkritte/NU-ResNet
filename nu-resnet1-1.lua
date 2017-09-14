--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local Transfer = nn.ELU
local Max = cudnn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   local function shortcut(nInputPlane, nOutputPlane,stride)
        return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane,1,1,stride,stride,0,0))
            :add(SBatchNorm(nOutputPlane))
   end
   
   local function resblock(input, output) --64,16
      res_output = output + (output*2) + output
	  local s = nn.Sequential()
	  s:add(nuinnet(input,output)) 
	  s:add(Convolution(res_output,res_output,1,1,1,1,0,0))
      s:add(SBatchNorm(res_output))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(input, res_output,1)))
         :add(nn.CAddTable(true))
         :add(Transfer(1.0,true))
    end
	
    function nuinnet(input,output)
	
	  local nu1 = nn.Sequential()
      nu1:add(Convolution(input,output,1,1,1,1,0,0))
      nu1:add(SBatchNorm(output))
      nu1:add(Transfer(1.0,true))
	  
	  local nu2 = nn.Sequential()  
	  nu2:add(Convolution(input,(output*1.5),1,1,1,1,0,0))
      nu2:add(SBatchNorm((output*1.5)))
      nu2:add(Transfer(1.0,true))
	  nu2:add(Convolution((output*1.5), (output*2), 3, 3, 1, 1, 1, 1))
      nu2:add(SBatchNorm((output*2)))
      nu2:add(Transfer(1.0,true))
	  
	  local nu3 = nn.Sequential()
	  nu3:add(Convolution(input, (output/4), 1, 1, 1, 1, 0, 0))
      nu3:add(SBatchNorm((output/4)))
      nu3:add(Transfer(1.0,true))
	  nu3:add(Convolution((output/4), (output/2), 3, 3, 1, 1, 1, 1))
      nu3:add(SBatchNorm((output/2)))
      nu3:add(Transfer(1.0,true))
	  nu3:add(Convolution((output/2), (output/2), 3, 3, 1, 1, 1, 1))
      nu3:add(SBatchNorm((output/2)))
      nu3:add(Transfer(1.0,true))
	  
	  local nu4 = nn.Sequential()
	  nu4:add(Convolution(input, (output/4), 1, 1, 1, 1, 0, 0))
      nu4:add(SBatchNorm((output/4)))
      nu4:add(Transfer(1.0,true))
	  nu4:add(Convolution((output/4), (output/2), 3, 3, 1, 1, 1, 1))
      nu4:add(SBatchNorm((output/2)))
      nu4:add(Transfer(1.0,true))
	  nu4:add(Convolution((output/2), (output/2), 3, 3, 1, 1, 1, 1))
      nu4:add(SBatchNorm((output/2)))
      nu4:add(Transfer(1.0,true))
	  nu4:add(Convolution((output/2), (output/2), 3, 3, 1, 1, 1, 1))
      nu4:add(SBatchNorm((output/2)))
      nu4:add(Transfer(1.0,true))
	  
	 local net = nn.Concat(2)
        net:add(nu1)
        net:add(nu2)
        net:add(nu3)
        net:add(nu4)
     return net

   end
   

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model
      --model:add(Convolution(3,3,1,1,1,1,0,0))
      --model:add(SBatchNorm(3))
      --model:add(ReLU(true))
      --model:add(slice(3))
      --model:add(nn.Identity())


      model:add(Convolution(3,64,5,5,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(Transfer(1.0,true))
      model:add(Max(3,3,2,2,1,1))
      model:add(resblock(64,16))
	  model:add(resblock(64,16))
	  model:add(Max(3,3,2,2,1,1))
	  model:add(resblock(64,32))
	  model:add(resblock(128,32))
	  model:add(Max(3,3,2,2,1,1))
	  model:add(resblock(128,64))
	  model:add(resblock(256,64))
	  model:add(Max(3,3,2,2,1,1))
	  model:add(resblock(256,128))
	  model:add(resblock(512,128))

      model:add(Convolution(512,1024,1,1,1,1,0,0))
      model:add(SBatchNorm(1024))
      model:add(Transfer(1.0,true))
      model:add(Avg(8, 8, 1, 1))

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
