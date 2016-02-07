%%%% tracking entry %%%%%

cellName='N2DL-HeLa';
dataset='train';
sq=2;
numFrame=92;
seqLength=6;

cellBlock=cell(1,seqLength);
str=sprintf('../data/%s/%s/%02d_SEG_DATA/data_%02d.mat',cellName,dataset,sq,1);
S=load(str);
cellBlock{seqLength}=S.segFrame;
for i=1:1:numel(cellBlock{seqLength})
    cellBlock{seqLength}{i}.id=i;
end
clear S

for i=2:1:numFrame
    for j=1:1:seqLength-1
        cellBlock{j}=cellBlock{j+1};
    end
    
    str=sprintf('../data/%s/%s/%02d_SEG_DATA/data_%02d.mat',cellName,dataset,sq,i);
    S=load(str);
    cellBlock{seqLength}=S.segFrame;
    
    cellBlock = new_EMD(cellBlock,i);

    cellBlock{seqLength} = segUpdate(cellBlock{seqLength-1}, cellBlock{seqLength});

    %%% save the tracking result %%%
    str=sprintf('../data/%s/%s/%02d_Track/track_%02d.mat',cellName,dataset,sq,i-1);
    cellFrame = cellBlock{seqLength-1};
    save(str,'cellFrame');
    clear cellFrame
end

