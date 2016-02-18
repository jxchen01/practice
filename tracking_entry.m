%%%% tracking entry %%%%%
cellName='N2DL-HeLa';
dataset='train';
sq=2;
numFrame=92;
seqLength=6;

opt=struct('cellName',cellName,'dataset',dataset,'sq',sq,'numFrame',...
    numFrame,'simpleMatchDist',5,'simpleMatchArea',10,...
    'maxMigration',60,'AcceptRateThreshold',0.55);

cellBlock=cell(1,seqLength);
str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,1);
S=load(str);
cellBlock{seqLength}=S.segFrame0;
maxTrack=numel(cellBlock{seqLength});
for i=1:1:maxTrack
    cellBlock{seqLength}{i}.id=i;
end
clear S

for i=2:1:numFrame
    for j=1:1:seqLength-1
        cellBlock{j}=cellBlock{j+1};
    end
    
    disp(i)
    
    %str=sprintf('../data/%s/%s/%02d_SEG_DATA/data_%02d.mat',cellName,dataset,sq,i);
    str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,i);
    S=load(str);
    cellBlock{seqLength}=S.segFrame0;
    clear S
    
    %%%% load raw image %%%%
    str=sprintf('../data/%s/%s/%02d/t%02d.tif',cellName,dataset,sq,i-1);
    I=mat2gray(imread(str));
    I=adapthisteq(I);
    
    disp('ready for EMD')
    cellBlock = new_EMD(cellBlock,i,opt);

    disp('start post processing')
    [cellBlock{seqLength-1},cellBlock{seqLength},maxTrack] = segUpdate(cellBlock{seqLength-1},...
        cellBlock{seqLength}, I,i,maxTrack);
    
    %%% save the tracking result %%%
    str=sprintf('../data/%s/%s/%02d_Track/track_%02d.mat',cellName,dataset,sq,i-1);
    cellFrame = cellBlock{seqLength-1};
    save(str,'cellFrame');
    clear cellFrame
end

