require 'rnn'
lfs=require('lfs')

version = 1

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a RNN Model on sample dataset using LSTM or GRU to compute state-values in EMD.')
cmd:text('Example:')
cmd:text("my-code.lua --cuda --useDevice 2 --progress --zeroFirst --cutoffNorm 4 --rho 10")
cmd:text('Options:')
cmd:option('--lr', 0.001, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 400, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.99, 'momentum')
cmd:option('--maxOutNorm', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 2, 'sets the device (GPU) to use')
cmd:option('--nIteration', 80000, 'maximum number of iteration to run')
cmd:option('--subIteration',5,'number of training steps in each subset')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

-- recurrent layer 
cmd:option('--lstm', false, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--gru', true, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('--rho', 6, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', true, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.1, 'probability of zeroing a neuron (dropout probability)')

-- file path
cmd:option('--fpath','/home/jchen16/code/Tracking_System/code/train','directory to data')
cmd:option('--trainType','gt','gt or seg')

cmd:text()
opt = cmd:parse(arg or {})

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
end

local fpath=opt.fpath
local trainType = opt.trainType

--[[Data]]--
numPredict=5

local datapath=string.format('%s/%s/%s',fpath,trainType,'data');
local targetpath=string.format('%s/%s/%s',fpath,trainType,'target');

print(datapath)
print(targetpath)

local iter_data, datadir = lfs.dir(datapath)
local datafile = datadir:next()
local f_data = datapath..'/'..datafile
local attr_data = lfs.attributes (f_data)
print(f_data)
while datafile == "." or datafile == ".." or  attr_data.mode == "directory" do
    datafile = datadir:next()
    f_data = datapath..'/'..datafile
    attr_data = lfs.attributes (f_data)
    print('w:  '..f_data)
end

local iter_target, targetdir = lfs.dir(targetpath)
local targetfile = targetdir:next()
local f_target = targetpath..'/'..targetfile
local attr_target = lfs.attributes(f_target)
while targetfile == "." or targetfile == ".." or  attr_target.mode == "directory" do
    targetfile = targetdir:next()
    f_target = targetpath..'/'..targetfile
    attr_target = lfs.attributes (f_target)
end

local data = torch.load(f_data)
local labels = torch.load(f_target)

COLS = data:size(2)
SEQS = labels:size(1)
ROWS = SEQS*6;

if opt.cuda then
  print('shipping data to cuda')
  data=data:cuda()
  labels=labels:cuda()
end
print('Good')
collectgarbage()

print('start to build model...')

--[[Model]]--

-- RNN model
lm = nn.Sequential()
local hiddenSize= {512,1024,1024,512}
local inputSize = 512

lm:add(nn.Sequencer(nn.Linear(COLS,inputSize)))

for i,hs in ipairs(hiddenSize) do
  
   -- recurrent layer
   local rnn
   if opt.gru then
      -- Gated Recurrent Units
      rnn = nn.GRU(inputSize, hs)
   elseif opt.lstm then
      -- Long Short Term Memory
      rnn = nn.FastLSTM(inputSize, hs)
   end

   lm:add(nn.Sequencer(rnn))
   
   if opt.dropout then -- dropout it applied between recurrent layers
      lm:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   end
   
   inputSize = hs
end

-- output layer
lm:add(nn.SelectTable(-1))
lm:add(nn.Linear(inputSize, numPredict))

if opt.uniform > 0 then
   for k,param in ipairs(lm:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

--criterion = nn.SequencerCriterion(nn.MSECriterion())
criterion = nn.MSECriterion()

-- linear decay
opt.decayFactor = (opt.minLR - opt.lr)/opt.saturateEpoch

print('model is done')

if opt.cuda then
   print('shipping model to cuda')
   lm:cuda()
   print('shippig criterion to cuda')
   criterion = criterion:cuda()
end

--[[Experiment]]--
offsets = torch.LongTensor(opt.batchSize):random(1,SEQS)

for k=1, opt.nIteration do 

    for innerK=1, opt.subIteration  do

      print(innerK)

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

 --   if k<10000 then
 --   	print('Iter: '.. k ..' Err: '.. err)
 --   end
  
    	lm:zeroGradParameters()

    	local gradOutputs = criterion:backward(outputs,targets)
    	local gradInputs = lm:backward(inputs,gradOutputs)
    
    	lm:updateParameters(opt.lr)

    	if innerK % 10 == 0 then collectgarbage() end 

    end

    print('done')
  
    if (k % 53 ==0) then
      print('Iter: '.. k ..' Err: '.. err)
      filename=string.format('%s/checkpoint/net_%f.bin',lfs.currentdir(),k);
      torch.save(filename,lm);
    end

    datafile = datadir:next()
    targetfile = targetdir:next()
    if (not datafile) or (not targetfile) then
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

    local datafile = datadir:next() 
    local targetfile = targetdir:next()

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

local inputs = {}

for step=1, opt.rho do
   inputs[step]= inputs[step] or data.new()
   inputs[step]:index(data,1,offsets)
   offsets:add(1)
end

local outputs=lm:forward(inputs)

torch.save('write.dat', outputs:float(),'ascii')
torch.save('write.bin', outputs:float())

--print(outputs)
--print(type(outputs))

--[[
local myfile = hdf5.open('write.h5','w')
myfile:write('path/to/data',outputs)
myfile:close()
--]]
