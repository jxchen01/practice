fpath='train_gt.csv'
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

print(i)

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


local outpath='data_gt.t7'
torch.save(outpath,data)
local outpath2='target_gt.t7'
torch.save(outpath2,target)
--[[
local input_data = torch.load(outpath)
local target_data = torch.load(outpath2)

print(data:size())
print(target:size())
--]]
