%%%%% build training data %%%%%
cellName='N2DL-HeLa';
dataset='train';
sq=2;
numFrame=92;
maxMigration=60;
seqLength=6;

%%% load mother-child relationship of mitosis %%%
str=sprintf('../data/%s/%s/%02d_GT/man_track.txt',cellName,dataset,sq);
rel=dlmread(str,' ');
rel=uint16(rel);
maxID=max(rel(:,1));
motherMap=zeros(maxID,1);
for i=1:1:size(rel,1)
    if(rel(i,4)>1e-5)
        motherMap(rel(i,1))=rel(i,4);
    end
end
    
% % part 1: purely based on ground truth
% cellBlock=cell(1,seqLength);
% str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,1);
% S=load(str);
% cellBlock{seqLength}=S.cellFrame0;
% 
% [dimx,dimy]=size(S.cellFrame0{1}.seg);
% clear S
% 
% fid=fopen('train_gt.csv','w');
% 
% for i=2:1:numFrame    
%     for j=1:1:seqLength-1
%         cellBlock{j}=cellBlock{j+1};
%     end
% 
%     str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,i);
%     S=load(str);
%     cellBlock{seqLength}=S.cellFrame0;
% 
%     for j=1:1:numel(cellBlock{seqLength})
%         c1=cellBlock{seqLength}{j}.Centroid;
%         cid=cellBlock{seqLength}{j}.id;
%         stat=regionprops(cellBlock{seqLength}{j}.seg,'Area',...
%             'MajorAxisLength','MinorAxisLength','Orientation');
%         topo1=[stat(1).Area,stat(1).MajorAxisLength,stat(1).MinorAxisLength,stat(1).Orientation];
% 
%         for k=1:1:numel(cellBlock{seqLength-1})
%             flag=0;
%             if(cellBlock{seqLength-1}{k}.id==cid)
%                 flag=1;
%                 cellid=k;
%             elseif(cellBlock{seqLength-1}{k}.id == motherMap(cid))
%                 flag=3;
%                 cellid=k;
%             else
%                 c2=cellBlock{seqLength-1}{k}.Centroid;
%                 if(norm(c1-c2)<maxMigration)
%                     flag=2;
%                     cellid=k;
%                 end
%             end
%             
%             if(flag>0)
%                 str=sprintf('../data/%s/%s/%02d_CELL_PATCH_OUT/%02d/%03d.tif.features',cellName,dataset,sq,i,j);
%                 M=dlmread(str,'');
%                 Mat=zeros(seqLength,M(1,1)+5);
%                 Mat(seqLength,1:M(1,1))=M(2,1:M(1,1));
%                 Mat(seqLength,end-5:end-2)=topo1(:);
%                 Mat(seqLength,end-1:end)=c1(:);
%                 
%                 for t=seqLength-1:-1:1
%                     if(cellid>0)
%                         c2=cellBlock{t}{cellid}.Centroid;
%                         stat=regionprops(cellBlock{t}{cellid}.seg,'Area',...
%                             'MajorAxisLength','MinorAxisLength','Orientation');
%                         topo2=[stat(1).Area,stat(1).MajorAxisLength,stat(1).MinorAxisLength,stat(1).Orientation];
%                         
%                         str=sprintf('../data/%s/%s/%02d_CELL_PATCH_OUT/%02d/%03d.tif.features',cellName,dataset,sq,t-seqLength+i,cellid);
%                         M=dlmread(str,'');
%                         Mat(t,1:M(1,1))=M(2,1:M(1,1));
%                         Mat(t,end-5:end-2)=topo2(:);
%                         Mat(t,end-1:end)=c2(:);
%                         
%                         if(~isempty(cellBlock{t}{cellid}.parent))
%                             cellid=cellBlock{t}{cellid}.parent;
%                         else
%                             pid=motherMap(cellBlock{t}{cellid}.id);
%                             if(t>1 && pid>0)
%                                 for pt=1:1:numel(cellBlock{t-1})
%                                     if(cellBlock{t-1}{pt}.id==pid)
%                                         cellid=pt;
%                                         break;
%                                     end
%                                 end
%                             else
%                                 cellid=-1;
%                             end
%                         end
%                     else
%                         Mat(t,:)=0;
%                     end
%                 end
%                 
%                 %%%% print to file %%%%
%                 for t=1:1:seqLength
%                     for w=1:1:size(Mat,2)-1
%                         fprintf(fid,'%f,',Mat(t,w));
%                     end
%                     fprintf(fid,'%f\n',Mat(t,end));
%                 end
%                 if(flag==1)
%                     fprintf(fid,'%f,%f,%f\n',0.1,1.0,1.0);
%                 elseif(flag==2)
%                     nc=nnz(motherMap==cellBlock{seqLength-1}{k}.id);
%                     if(nc>1 )
%                         fprintf(fid,'%f,%f,%f\n',1.0,1.0,1.0);
%                     else
%                         fprintf(fid,'%f,%f,%f\n',0.85,1.0,1.0);
%                     end              
%                 elseif(flag==3)
%                     fprintf(fid,'%f,%f,%f\n',0.01,1.0/nnz(motherMap==motherMap(cid)),1.0);
%                 else
%                     error('error in flag');
%                 end
%             end
%         end
%     end
% end
% fclose(fid);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% part 2: based on segmentation and ground truth history
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cellBlock=cell(1,seqLength);
fid=fopen('train_seg.csv','w');

str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,1);
S=load(str);

for i=2:1:numFrame
    
    for j=1:1:seqLength-2
        cellBlock{j}=cellBlock{j+1};
    end

    %%%% S has been loaded in the previous iteration %%%
    cellBlock{seqLength-1}=S.cellFrame0;
    clear S
    
    str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,i);
    S=load(str);
    cellBlock{seqLength}=S.segFrame0;   

    for j=1:1:numel(cellBlock{seqLength})
        c1=cellBlock{seqLength}{j}.Centroid;
        cid=cellBlock{seqLength}{j}.id;
        stat=regionprops(cellBlock{seqLength}{j}.seg,'Area',...
            'MajorAxisLength','MinorAxisLength','Orientation');
        topo1=[stat(1).Area,stat(1).MajorAxisLength,stat(1).MinorAxisLength,stat(1).Orientation];

        for k=1:1:numel(cellBlock{seqLength-1})
            
            %%%% determine the relationship %%%%%
            flag=0;
            if(any(cid==cellBlock{seqLength-1}{k}.id)) % migration
                flag=1;
                cellid=k;
            else
                for t=1:1:numel(cid)
                    if(motherMap(cid(t))==cellBlock{seqLength-1}{k}.id)
                        flag=3;
                        break;
                    end
                end
                if(flag>0)
                    cellid=k;
                else
                    c2=cellBlock{seqLength-1}{k}.Centroid;
                    if(norm(c1-c2)<maxMigration)
                        flag=2;
                        cellid=k;
                    end
                end
            end
            
            if(flag>0)
                str=sprintf('../data/%s/%s/%02d_SEG_PATCH_OUT/%02d/%03d.tif.features',cellName,dataset,sq,i,j);
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
                        
                        str=sprintf('../data/%s/%s/%02d_CELL_PATCH_OUT/%02d/%03d.tif.features',cellName,dataset,sq,t-seqLength+i,cellid);
                        M=dlmread(str,'');
                        Mat(t,1:M(1,1))=M(2,1:M(1,1));
                        Mat(t,end-5:end-2)=topo2(:);
                        Mat(t,end-1:end)=c2(:);
                        
                        if(~isempty(cellBlock{t}{cellid}.parent))
                            cellid=cellBlock{t}{cellid}.parent;
                        else
                            pid=motherMap(cellBlock{t}{cellid}.id);
                            if(t>1 && pid>0)
                                for pt=1:1:numel(cellBlock{t-1})
                                    if(cellBlock{t-1}{pt}.id==pid)
                                        cellid=pt;
                                        break;
                                    end
                                end     
                            else
                                cellid=-1;
                            end
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
                if(flag==1) % migration
                    fprintf(fid,'%f,%f,%f\n',0.1,1.0,1.0/numel(cid));
                elseif(flag==2) % false connection
                    nc=nnz(motherMap==cellBlock{seqLength-1}{k}.id);
                    if(nc>1)
                        fprintf(fid,'%f,%f,%f\n',1.0,1.0,1.0);
                    else
                        fprintf(fid,'%f,%f,%f\n',0.85,1.0,1.0);
                    end    
                elseif(flag==3) % mitosis
                    count=0;
                    for t=1:1:numel(cid)
                        if(motherMap(cid(t))==cellBlock{seqLength-1}{k}.id)
                            count=count+1;
                        end
                    end
                    fprintf(fid,'%f,%f,%f\n',0.01,...
                        double(count)/double(nnz(motherMap==cellBlock{seqLength-1}{k}.id)),double(count)/double(numel(cid)));
                    clear count
                else
                    error('error in flag');
                end
            end
        end
    end
end
fclose(fid);

