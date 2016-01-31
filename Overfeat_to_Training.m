%%%%% build training data %%%%%
cellName='N2DL-HeLa';
dataset='train';
sq=2;
numFrame=92;
maxMigration=60;
seqLength=6;

% part 1: purely based on ground truth
cellBlock=cell(1,seqLength);
str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,1);
S=load(str);
cellBlock{seqLength}=S.cellFrame0;

[dimx,dimy]=size(S.cellFrame0{1}.seg);
clear S

fid=fopen('train_gt.txt','w');

offset=-seqLength+1;
for i=2:1:numFrame
    offset=offset+1;
    
    for j=1:1:seqLength-1
        cellBlock{j}=cellBlock{j+1};
    end

    str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,offset+seqLength);
    S=load(str);
    cellBlock{seqLength}=S.cellFrame0;

    for j=1:1:numel(cellBlock{seqLength})
        c1=cellBlock{seqLength}{j}.Centroid;
        cid=cellBlock{seqLength}{j}.id;
        stat=regionprops(cellBlock{seqLength}{j}.seg,'Area',...
            'MajorAxisLength','MinorAxisLength','Orientation');
        topo1=[stat(1).Area,stat(1).MajorAxisLength,stat(1).MinorAxisLength,stat(1).Orientation];
%         reg1=cellBlock{seqLength}{j}.seg;
%         bd=bwboundaries(reg1);
%         L1=bd{1};
%         p1=reshape(L1',1,numel(L1));
        for k=1:1:numel(cellBlock{seqLength-1})
            flag=0;
            if(cellBlock{seqLength-1}{k}.id==cid)
                flag=1;
                cellid=k;
            else
                c2=cellBlock{seqLength-1}{k}.Centroid;
                if(norm(c1-c2)<maxMigration)
                    flag=2;
                    cellid=k;
                end
            end
            
            if(flag>0)
                str=sprintf('../data/%s/%s/%02d_CELL_PATCH_OUT/%02d/%03d.tif.features',cellName,dataset,sq,seqLength+offset,j);
                M=dlmread(str,'');
                Mat=zeros(seqLength,M(1,1)+5);
                Mat(seqLength,1:M(1,1))=M(2,1:M(1,1));
                Mat(seqLength,end-5:end-2)=topo1(:);
                Mat(seqLength,end-1:end)=c1(:);
                
                for t=seqLength-1:-1:1
                    if(cellid>0)
                        c2=cellBlock{t}{cellid}.Centroid;
                        stat=regionprops(cellBlock{t}{cellid}.seg,'Area',...
                            'MajorAxisLength','MinorAxisLength','Orientation');
                        topo2=[stat(1).Area,stat(1).MajorAxisLength,stat(1).MinorAxisLength,stat(1).Orientation];
                        
                        str=sprintf('../data/%s/%s/%02d_CELL_PATCH_OUT/%02d/%03d.tif.features',cellName,dataset,sq,t+offset,cellid);
                        M=dlmread(str,'');
                        Mat(t,1:M(1,1))=M(2,1:M(1,1));
                        Mat(t,end-5:end-2)=topo2(:);
                        Mat(t,end-1:end)=c2(:);
                        
                        if(~isempty(cellBlock{t}{cellid}.parent))
                            cellid=cellBlock{t}{cellid}.parent;
                        else
                            cellid=-1;
                        end
                    else
                        Mat(t,:)=0;
                    end
                end
                
                %%%% print to file %%%%
                for t=1:1:seqLength
                    for w=1:1:size(Mat,2)-1
                        fprintf(fid,'%f,',Mat(t,w));
                    end
                    fprintf(fid,'%f\n',Mat(t,end));
                end
                if(flag==1)
                    fprintf(fid,'%f,%f,%f\n',0.01,1.0,1/numel(cellBlock{seqLength-1}{k}.child));
                elseif(flag==2)
                    fprintf(fid,'%f,%f,%f\n',0.99,1.0,1.0);
                else
                    error('error in flag');
                end
            end
        end
    end
end

% part 2: based on my segmentation and ground truth history



