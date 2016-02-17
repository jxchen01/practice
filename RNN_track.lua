require 'rnn'
--lfs=require('lfs')
matio=require('matio')

version = 1

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a RNN Model on sample dataset using LSTM or GRU to compute state-values in EMD.')
cmd:text('Example:')
cmd:text("my-code.lua --cuda --useDevice 2 --progress --zeroFirst --cutoffNorm 4 --rho 10")
cmd:text('Options:')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 2, 'sets the device (GPU) to use')

-- recurrent layer 
cmd:option('--rho', 6, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--frame',1,'frame being processed')

-- file path
cmd:option('--fpath','/home/jchen16/code/Tracking_System/code/train/seg/data/data_seg_1.t7','directory to data')
cmd:option('--netpath','/home/jchen16/code/Tracking_System/code/checkpoint/seg/net_3650.000000.bin','directory to model')

cmd:text()
opt = cmd:parse(arg or {})

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
end

local f_data=opt.fpath

--[[Data]]--
numPredict=5

local data = torch.load(f_data)

COLS = data:size(2)
ROWS = data:size(1)
SEQS = ROWS/opt.rho

if opt.cuda then
  print('shipping data to cuda')
  data=data:cuda()
end
collectgarbage()

--[[Model]]--

-- RNN model
lm = torch.load(opt.netpath);

print('model is loaded')

--[[Results]]--
local total=0;
local offsets = torch.LongTensor(opt.batchSize)
local results=torch.FloatTensor(SEQS+opt.batchSize,numPredict)
i=0; offsets:apply(function() i = i + 1; return i end)

while total<SEQS do

  indices = offsets:clone()
  indices:add(-1)
  indices:mul(opt.rho)
  indices:add(1)

  local inputs = {}

  for step=1, opt.rho do
    inputs[step]= inputs[step] or data.new()
    inputs[step]:index(data,1,offsets)
    indices:add(1)
  end

  local outputs=lm:forward(inputs)
  local foutput=outputs:float()

  for tt=1,opt.batchSize do
    for p=1,numPredict do
      results[offsets[tt]][p]=foutput[tt][p]
    end
  end

  total=total+opt.batchSize
  for tt=1,opt.batchSize do
    offsets:add(1)
    offsets[offsets:gt(SEQS)]=1
  end
  collectgarbage()
end

outname=string.format('/home/jchen16/code/Tracking_System/code/RNN_result/results_%d,mat',opt.frame)
matio.save(outname,results)

