lfs=require('lfs')

trainType='seg'
fpath=string.format('/home/jchen16/code/Tracking_System/code/train/%s/',trainType);

numPredict=5;

print(fpath)

iter=1;
for file in lfs.dir( fpath ) do
    if file ~= "." and file ~= ".." then
        -- File is the current file or directory name
        print( "Found file: " .. file )

        local f = fpath..'/'..file
        local attr = lfs.attributes (f)
    	assert (type(attr) == "table")

        if attr.mode ~= "directory" then

            if iter==1 then 
	    	local i=0
     	    	for line in io.lines(f) do 
    		    if i==0 then
		    	COLS = #line:split(',')
    		    end
      	 	    i=i+1
	    	end

	    	SEQS=math.ceil(i/7);
	    	ROWS=SEQS*6;

	    	--print(SEQS)
	    	--print(ROWS)
	    end
    
	    local data=torch.Tensor(ROWS,COLS)
    	local labels=torch.Tensor(SEQS,numPredict)
    	local i,j,k,t=0,0,0,0
    
    	for line in io.lines(f) do
            i=i+1
            local l=line:split(',')
            if math.fmod(i,7)==0 then
            	j=j+1
            	for key, val in ipairs(l) do
                    labels[j][key]=val
            	end
                local x0=data[k][COLS-5]
                local y0=data[k][COLS-4]
                for t=1,6 do
                    data[k-t+1][COLS-5]=data[k-t+1][COLS-5]-x0
                    data[k-t+1][COLS-4]=data[k-t+1][COLS-4]-y0
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
  
    	iter = iter+1
        
     end
end

