%%%%% build training data %%%%%
maxMigration=60;
seqLength=6;


% part 1: purely based on ground truth
cellBlock=cell(1,seqLength);
str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,1);
S=load(str);
cellBlock{seqLength}=S.cellFrame0;
clear S

for i=2:1:numFrame
    for j=1:1:seqLength-1
        cellBlock{j}=cellBlock{j+1};
    end

    str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,i);
    S=load(str);
    cellBlock{seqLength}=S.cellFrame0;

    for j=1:1:numel(cellBlock{seqLength})
        c1=cellBlock{seqLength}{j}.Centroid;
        cid=cellBlock{seqLength}{j}.id;
        for k=1:1:numel(cellBlock{seqLength-1})
            flag=0;
            if(cellBlock{seqLength-1}{k}.id==cid)
                flag=1;
            else
                c2=cellBlock{seqLength-1}{k}.Centroid;
                if(norm(c1-c2)<maxMigration)
                    flag=2;
                end
            end
            
            if(flag>0)
                
            end
        end
    end
end

% part 2: based on my segmentation and ground truth history



