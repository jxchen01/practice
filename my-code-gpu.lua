require 'rnn'
version = 1

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a RNN Model on sample dataset using LSTM or GRU to compute state-values in EMD.')
cmd:text('Example:')
cmd:text("my-code.lua --cuda --useDevice 2 --progress --zeroFirst --cutoffNorm 4 --rho 10")
cmd:text('Options:')
cmd:option('--lr', 0.01, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 400, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 4, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--nIteration', 2000, 'maximum number of iteration to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

-- recurrent layer 
cmd:option('--lstm', false, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--gru', true, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('--rho', 6, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropout', false, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.2, 'probability of zeroing a neuron (dropout probability)')

cmd:text()
opt = cmd:parse(arg or {})

--[[Data]]--
fpath='train_2.csv'
numPredict=3;

local i=0
for line in io.lines(fpath) do
    if i==0 then
        COLS = #line:split(',')
    end
    i=i+1
end

SEQS=math.ceil(i/7);
ROWS=SEQS*6;

local data=torch.Tensor(ROWS,COLS)
local labels=torch.Tensor(SEQS,numPredict)
local i=0
local j=0
local k=0
for line in io.lines(fpath) do
    i=i+1
    local l=line:split(',')
    if math.fmod(i,7)==0 then
        j=j+1
        for key, val in ipairs(l) do
            labels[j][key]=val
        end
    else
        k=k+1
        for key, val in ipairs(l) do
            data[k][key]=val
        end
    end
end

--[[Model]]--

-- language model
lm = nn.Sequential()
local hiddenSize= {200,200}
local inputSize = 200

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
-- lm:add(nn.Sequencer(nn.Linear(inputSize, numPredict)))

lm:add(nn.SelectTable(-1))
lm:add(nn.Linear(inputSize, numPredict))

if opt.uniform > 0 then
   for k,param in ipairs(lm:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end

-- will recurse a single continuous sequence
-- lm:remember((opt.lstm or opt.gru) and 'both' or 'eval')

--criterion = nn.SequencerCriterion(nn.MSECriterion())
criterion = nn.MSECriterion()

-- linear decay
opt.decayFactor = (opt.minLR - opt.lr)/opt.saturateEpoch

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   lm:cuda()
   print('shipping model to cuda')
   data=data:cuda()
   labels=labels:cuda()
   print('shipping data to cuda')
   criterion = criterion:cuda()
   print('shippig criterion to cuda')
end

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

   -- rnn:zeroGradParameters() 

   
    local outputs = lm:forward(inputs)
    

    local err = criterion:forward(outputs:float(),targets:float())

    print('Iter: '.. k ..' Err: '.. err)

    lm:zeroGradParameters()

    local gradOutputs = criterion:backward(outputs,targets)
    local gradInputs = lm:backward(inputs,gradOutputs)
    
    lm:updateParameters(opt.lr)
end


local inputs = {}

for step=1, opt.rho do
   inputs[step]= inputs[step] or data.new()
   inputs[step]:index(data,1,offsets)
   offsets:add(1)
end

local outputs=lm:forward(inputs)

torch.save('write.dat', outputs:float(),'ascii')

print(outputs)
print(type(outputs))

--[[
local myfile = hdf5.open('write.h5','w')
myfile:write('path/to/data',outputs)
myfile:close()
--]]
