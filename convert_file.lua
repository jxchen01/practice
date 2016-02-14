require 'luafilesystem'

trainType='gt'
fpath=string.format('./train/%s/',trainType);

numPredict=5;

local dir = lfs.dir(fpath)
local file = dir:next() 

local i=0
for line in io.lines(file) do 
    if i==0 then
	COLS = #line:split(',')
    end
    i=i+1
end

SEQS=math.ceil(i/7);
ROWS=SEQS*6;

print(SEQS)
print(ROWS)

iter=1;
while file do

    local data=torch.Tensor(ROWS,COLS)
    local labels=torch.Tensor(SEQS,numPredict)
    local i=0
    local j=0
    local k=0
    for line in io.lines(file) do
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


    local outpath=string.format('%s%s/data_%s_%d.t7',fpath,'data',trainType,iter)
    torch.save(outpath,data)
    local outpath2=string.format('%s%s/target_%s_%d.t7',fpath,'target',trainType,iter)
    torch.save(outpath2,labels)
  
    file = dir:next()
    iter = iter+1
end
dir:close()
--[[
local input_data = torch.load(outpath)
local target_data = torch.load(outpath2)

print(data:size())
print(target:size())
--]]
