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


for k=1, opt.nIteration do 

    for innerK=1, opt.subIteration  do

      local inputs = {}

    	indices = offsets:clone()
    	indices:add(-1)
    	indices:mul(opt.rho)
    	indices:add(1)
   
   	  -- local inputs=torch.LongTensor(opt.rho,opt.batchSize,COLS)
    	for step=1, opt.rho do
        inputs[step]= inputs[step] or data.new()
   	    inputs[step]:index(data,1,indices)
        indices:add(1)
      end

    	--local targets=torch.LongTensor(opt.rho,numPredict)
    	targets = labels.new()
    	targets:index(labels,1,offsets)

    	offsets:add(1)
    	offsets[offsets:gt(SEQS)]=1

    	local outputs = lm:forward(inputs)    

    	local err = criterion:forward(outputs:float(),targets:float())

      print('Iter: '.. k .. '  Inner: '.. innerK.. ' Err: '.. err)
        
    	lm:zeroGradParameters()

    	local gradOutputs = criterion:backward(outputs,targets)
    	local gradInputs = lm:backward(inputs,gradOutputs)
    
    	lm:updateParameters(opt.lr)

    	if innerK % 10 == 0 then collectgarbage() end 

    end

  
    if (k % 50 ==0) then
      --print('Iter: '.. k ..' Err: '.. err)
      filename=string.format('%s/checkpoint/net_%f.bin',lfs.currentdir(),k);
      torch.save(filename,lm);
    end

    datafile = datadir:next()
    targetfile = targetdir:next()
    if (not datafile) or (not targetfile) then
      assert(not pcall(datadir.next, datadir))
      assert(not pcall(targetdir.next, targetdir))
      iter_data, datadir = lfs.dir(datapath)
      iter_target, targetdir = lfs.dir(targetpath)
      datafile = datadir:next()
      targetfile = targetdir:next()
    end

    f_target = targetpath..'/'..targetfile
    attr_target = lfs.attributes(f_target)
    while targetfile == "." or targetfile == ".." or  attr_target.mode == "directory" do
      targetfile = targetdir:next()
      f_target = targetpath..'/'..targetfile
      attr_target = lfs.attributes (f_target)
    end

    f_data = datapath..'/'..datafile
    attr_data = lfs.attributes (f_data)
    while datafile == "." or datafile == ".." or  attr_data.mode == "directory" do
      datafile = datadir:next()
      f_data = datapath..'/'..datafile
      attr_data = lfs.attributes (f_data)
    end

    data = torch.load(f_data)
    labels = torch.load(f_target)

    COLS = data:size(2)
    SEQS = labels:size(1)
    ROWS = SEQS*6;

    if opt.cuda then
      data=data:cuda()
      labels=labels:cuda() 
    end

    offsets = torch.LongTensor(opt.batchSize):random(1,SEQS)
    collectgarbage()
    
end

datadir:close()
targetdir:close()

local total=0;
local offsets = torch.LongTensor(opt.batchSize)
local results=torch.LongTensor(SEQS+opt.batchSize,numPredict)
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

  for key, val in ipairs(offsets) do
    for p=1,numPredict do
      results[val][p]=outputs[key][p]:float()
    end
  end

  total=total+opt.batchSize
  for tt=1,opt.batchSize do
    offsets:add(1)
    offsets[offsets:gt(SEQS)]=1
  end
  print(total)
  collectgarbage()
end

matio.save('results.mat',results)

