--lfs=require('lfs')

cmd = torch.CmdLine()
cmd:option('--name','frame_1', 'data file')
opt = cmd:parse(arg or {})

trainType='seg'
f=string.format('/home/jchen16/code/Tracking_System/code/RNN_data/%s.csv',opt.name)

local attr = lfs.attributes (f)
assert (type(attr) == "table")
local i=0
for line in io.lines(f) do 
    if i==0 then
        COLS = #line:split(',')
    end
    i=i+1
end
ROWS=i;
SEQS=i/6;

local data=torch.Tensor(ROWS,COLS)
local i,k=0,0
    
for line in io.lines(f) do
    i=i+1
    local l=line:split(',')
        k=k+1
        for key, val in ipairs(l) do
            data[k][key]=val
        end
    end
end

local outpath=string.format('/home/jchen16/code/Tracking_System/code/RNN_data/%s.t7',opt.name)
torch.save(outpath,data)