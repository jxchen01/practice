require 'rnn'
require 'luafilesystem'

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
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--nIteration', 10000, 'maximum number of iteration to run')
cmd:option('--subIteration',500,'number of training steps in each subset')
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
cmd:option('--fPath','./train','directory to data')
cmd:option('--trainingType','gt','gt or seg')

cmd:text()
opt = cmd:parse(arg or {})

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
end

print('start to build model...')

--[[Model]]--

-- RNN model
lm = nn.Sequential()
local hiddenSize= {512,512}
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


--[[Data]]--
numPredict=5;

local datapath=string.format('%s/%s/%s/'opt.fpath,opt.trainType,'data');
local targetpath=string.format('%s/%s/%s/'opt.fpath,opt.trainType,'target');

local datadir = lfs.dir(datapath)
local datafile = datadir:next() 

local targetdir = lfs.dir(targetpath)
local targetfile = targetdir:next()

local data = torch.load(datafile)
local labels = torch.load(targetfile)

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

--[[Experiment]]--
offsets = torch.LongTensor(opt.batchSize):random(1,SEQS)

for k=1, opt.nIteration do 

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

  if(k<10000)
    print('Iter: '.. k ..' Err: '.. err)
  end
  
  lm:zeroGradParameters()

  local gradOutputs = criterion:backward(outputs,targets)
  local gradInputs = lm:backward(inputs,gradOutputs)
    
  lm:updateParameters(opt.lr)

  if (k>10000 and k % 1000 ==0) or k==opt.nIteration
      print('Iter: '.. k ..' Err: '.. err)
      filename=string.format('./checkpoint/net_%f.bin',k);
      torch.save(filename,lm);
  end 

  if k % opt.subIteration ==0 then
    local datafile = datadir:next() 
    local targetfile = targetdir:next()

    if (not datafile) or (not targetfile) then
      datadir:close()
      targetdir:close()

      local datadir = lfs.dir(datapath)
      local datafile = datadir:next() 

      local targetdir = lfs.dir(targetpath)
      local targetfile = targetdir:next()
    end

    local data = torch.load(datafile)
    local labels = torch.load(targetfile)

    COLS = data:size(2)
    SEQS = labels:size(1)
    ROWS = SEQS*6;

    if opt.cuda then
      data=data:cuda()
      labels=labels:cuda() 
    end

    offsets = torch.LongTensor(opt.batchSize):random(1,SEQS)
  end

  if k % 10 == 0 then collectgarbage() end
end

datadir:close()
targetdir:close()

-- torch.save('net.bin', lm)

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
