function [srcCellList,tarCellList]=new_EMD(cellBlock,frameID,motherMap,opt)        
% 
% %%%%%%% parameters %%%%%%%
opt.simpleMatchDist=5;
opt.simpleMatchArea=10;
opt.maxMigration=90;
opt.AcceptRateThreshold=0.55;
% minValidFlow=3;
% %halfROIs = algOptions.halfROIs;
% BoundThreshold = algOptions.BoundThresh;
% candiRadius=algOptions.candiRadius;
% bodyRatio=algOptions.bodyRatio;

% 
% % initialization
seqLength=numel(cellBlock);
srcNum = length(cellBlock{seqLength-1});
tarNum = length(cellBlock{seqLength});

% 
% imgSize = size(srcMat);
% dimx=imgSize(1);dimy=imgSize(2);
% 
% if(size(srcMat)~=size(tarMat))
%     error('error in local EMD');
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% compute the distance between signitures %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%% pre-check, if some relationship has been built %%%%
% for i=1:1:srcNum
%     if(~isempty(srcCellList{i}.child))
%         cid = srcCellList{i}.child;
%         if(numel(cid)>1)
%             disp('invalid pre-relationship')
%             keyboard
%         end
%         costMat(i,:)=-1;
%         costMat(:,cid)=-1;
%         uMat(i,:)=0; vMat(i,:)=0;
%         uMat(:,cid)=0; vMat(:,cid)=0;
%     end
% end

% compute matching cost and scaling parameters

costMat=zeros(srcNum+1,tarNum+1);
uMat=zeros(srcNum+1,tarNum+1);
vMat=zeros(srcNum+1,tarNum+1);

costMatIdx=zeros(srcNum,tarNum);
counter=0;
filename=['./RNN_data/frame_',num2str(frameID),'.csv'];
fid=fopen(filename,'w');
for j=1:1:tarNum
    c1=cellBlock{seqLength}{j}.Centroid;
    for k=1:1:srcNum

        c2=cellBlock{seqLength-1}{k}.Centroid;
        if(norm(c1-c2)>opt.maxMigration)
            continue;
        elseif(norm(c1-c2)<opt.simpleMatchDist && abs(cellBlock{seqLength}{j}.props(1)...
                -cellBlock{seqLength-1}{k}.props(1))<opt.simpleMatchArea)
            cellBlock{seqLength-1}{k}.child = j;
            cellBlock{seqLength}{j}.parent = k;
            costMat(k,:)=-1;
            costMat(:,j)=-1;
            uMat(k,:)=0; vMat(k,:)=0;
            uMat(:,j)=0; vMat(:,j)=0;
            continue;
        end
        
        counter=counter+1;
        costMatIdx(k,j)=counter; 
        
        str=sprintf('../data/%s/%s/%02d_CELL_PATCH_OUT/%02d/%03d.tif.features',cellName,dataset,sq,frameID,j);
        M=dlmread(str,'');
        Mat=zeros(seqLength,M(1,1)+5);
        Mat(seqLength,1:M(1,1))=M(2,1:M(1,1));
        Mat(seqLength,end-5:end-2)=cellBlock{seqLength}{j}.props(:);
        Mat(seqLength,end-1:end)=c1(:);
        
        for t=seqLength-1:-1:1
            if(cellid>0)
                c2=cellBlock{t}{cellid}.Centroid;
                
                str=sprintf('../data/%s/%s/%02d_CELL_PATCH_OUT/%02d/%03d.tif.features',cellName,dataset,sq,t-seqLength+frameID,cellid);
                M=dlmread(str,'');
                Mat(t,1:M(1,1))=M(2,1:M(1,1));
                Mat(t,end-5:end-2)=cellBlock{t}{cellid}.props(:);
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

    end
end
fclose(fid);

%%%% invoke RNN %%%%
cm1=sprintf('convert_frame.lua --name ''frame_%d''',frameID);
system(cm1)
RNN_name=['frame_',num2str(frameID),'.t7'];
cmm=['th RNN_track.lua --useDevice 2 --fPath %s ',RNN_name];
system(cmm)

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% fetch RNN results %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
rnnResult=importdata(['./RNN_result/results_',num2str(frameID),'.mat']);
leavingCost=zeros(srcNum,tarNum);
enteringCost=zeros(srcNum,tarNum);
for j=1:1:tarNum
    for k=1:1:srcNum
        if(costMat(k,j)>-1e-5)
            costMat(k,j)=rnnResult(costMatIdx(k,j),1);
            uMat(k,j)=rnnResult(costMatIdx(k,j),2);
            vMat(k,j)=rnnResult(costMatIdx(k,j),3);
            leavingCost(k,j)=rnnResult(costMatIdx(k,j),4);
            enteringCost(k,j)=rnnResult(costMatIdx(k,j),5);
        end
    end
end
for j=1:1:tarNum
    pv=enteringCost(:,j);
    pv=nonzeros(pv);
    costMat(end,j)=mean(pv);
end
for k=1:1:srcNum
    pv=leavingCost(k,:);
    pv=nonzeros(pv);
    costMat(k,end)=mean(pv);
end

costMat(srcNum+1,tarNum+1)=-1;

uMat(:,end)=1;
uMat(end,:)=1;
uMat(end,end)=0;

vMat(:,end)=1;
vMat(end,:)=1;
vMat(end,end)=0;

% %%% toy example %%%%
% srcNum=3; tarNum=3;
% costMat = [0.1, 0.6, 0.9, 0.85;
%            0.1, 0.9, 0.7, 0.98;
%            0.4, 0.1, 0.1, 1.00;
%            0.8, 0.99,1.0, -1];
% uMat = [1, 1, 1, 1;
%         1, 1, 1, 1;
%         1, 0.5,0.5,1;
%         1, 1, 1, 0];
% vMat = [0.5, 1, 1, 1;
%         0.5, 1, 1, 1;
%         1,   1, 1, 1;
%         1,   1, 1, 0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% perform matching on the whole region %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

good_idx = find(costMat>=0);
numVar=length(good_idx);
% build the linear optimization problem
weight=costMat(good_idx);

%Aeq = zeros(srcNum+tarNum, numVar);  % flow constraint at each node
%beq = ones(srcNum+tarNum, 1);
Aeq=[];

lb = zeros(numVar,1);
ub = ones(numVar,1);

% update Aeq
sz=[srcNum+1,tarNum+1];
for i=1:1:srcNum
    tidx = find(costMat(i,:)>=0);
    if(numel(tidx)==0)
        continue;
    end
    
    tmpRow=zeros(1,numVar);
    ind = sub2ind(sz,i.*ones(1,length(tidx)), tidx);
    
    for kk=1:1:numel(ind)
        ki = find(good_idx == ind(kk));
        if(~ki)
            disp('error');
            keyboard
        else
            tmpRow(1,ki)=uMat(ind(kk));
        end
    end
    Aeq = cat(1,Aeq,tmpRow);
end

for j=1:1:tarNum
    tidx = find(costMat(:,j)>=0);
    if(numel(tidx)==0)
        continue;
    end
    
    ind = sub2ind(sz,tidx,j.*ones(length(tidx),1));
    tmpRow = zeros(1,numVar);
    
    for kk=1:1:numel(ind)
        ki = find(good_idx == ind(kk));
        if(~ki)
            disp('error');
            keyboard
        else
            tmpRow(1,ki)=vMat(ind(kk));
        end
    end
    
    Aeq = cat(1,Aeq,tmpRow);
end

numConstraint = size(Aeq,1);
beq=ones(numConstraint,1);

if(isempty(weight) || isempty(beq) || isempty(Aeq))
    return
end

[xval,~,exitflag,output] = linprog(weight,[],[],Aeq,beq,lb,ub,[], options);
if(exitflag~=1)
    disp(output.message);
    error('error in EMD optimization');
end

%%% feasible matching
matchMat=zeros(srcNum+1,tarNum+1);
for i=1:1:numVar
    if(xval(i)>opt.AcceptRateThreshold)
        matchMat(good_idx(i))=1;
    end
end

for i=1:1:srcNum
    if(isempty(cellBlock{seqLength-1}{i}.child))
        ind=find(matchMat(i,1:end-1)>0.5);
        cellBlock{seqLength-1}{i}.child = ind;
    end
end

for j=1:1:tarNum
    if(isempty(cellBlock{seqLength}{j}.parent))
        ind=find(matchMat(1:end-1,j)>0.5);
        cellBlock{seqLength}{j}.parent = ind;
    end
end

    