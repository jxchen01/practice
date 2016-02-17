function [srcList, tarList,maxTrack] = segUpdate(srcList, tarList,tarImg,frameIdx,maxTrack)

patchSize=101; halfPatch=(patchSize-1)/2+1;
overFeatSize=231;
cellName='N2DL-HeLa';
dataset='train';
sq=2;

str_seg_patch=sprintf('../data/%s/%s/%02d_SEG_PATCH/%02d',cellName,dataset,sq,frameIdx);
overfeat_path='/home/jchen16/overfeat/src/overfeat';
tmp_path='./tmp_overfeat/';
feature_path=sprintf('../data/%s/%s/%02d_SEG_PATCH_OUT/',cellName,dataset,sq);

se=strel('disk',1);
se5=strel('disk',5,0);
sz=size(srcList{1}.seg);
dimx=sz(1); dimy=sz(2);

tarNum=numel(tarList);
numNewCell=0;

%%%%% get the max patch id %%%%%
maxID=0;
for i=1:1:tarNum
    if(tarList{i}.patch>maxID)
        maxID=tarList{i}.patch;
    end
end


for i=1:1:tarNum
    if(numel(tarList{i}.parent)>1)
        numCell = numel(tarList{i}.parent);
        pid=tarList{i}.parent;
        
        %%%% false merge %%%%
        bw=tarList{i}.seg;
        bw=imerode(bw,se);
        
        lab = cutRegion(bw,numCell);
        
        %%%% classic kmeans %%%%
%         ind=find(bw);
%         [xx,yy]=ind2sub(sz,ind);
%         idx = kmeans([xx,yy],numCell);
%         lab=zeros(sz);
%         for j=1:1:numel(idx)
%             lab(xx(j),yy(j))=idx(j);
%         end
%         clear ind xx yy bw
        
        %%%% replace i-th element %%%%%
        bw=ismember(lab,1);
        cc=bwconncomp(bw);
        if(cc.NumObjects>1)
            bw=zeros(sz);
            numPixels = cellfun(@numel,cc.PixelIdxList);
            [~,biggest_idx] = max(numPixels);
            bw(cc.PixelIdxList{biggest_idx}) = 1;
        end
        im_region = tarImg;
        mask = imdilate(bw,se5);
        im_region(~mask)=0;
        a = regionprops(bw,'Centroid','Area','MajorAxisLength','MinorAxisLength','Orientation');
        x0=round(a.Centroid(2));y0=round(a.Centroid(1));
        
        %%% create image patch %%%
        tmp=zeros(patchSize,patchSize);
        h=min([x0-1,y0-1,20,dimx-x0,dimy-y0]);
        tmp(halfPatch-h:1:halfPatch+h, halfPatch-h:1:halfPatch+h)=...
            im_region(x0-h:1:x0+h,y0-h:1:y0+h);
        tmp=imresize(tmp,[overFeatSize,overFeatSize]);
        tmp=mat2gray(tmp);
        
        numNewCell=numNewCell+1;
        rgb=cat(3,tmp,tmp,tmp);
        str_patch=sprintf('%s/%03d.tif',str_seg_patch,maxID+numNewCell);
        imwrite(rgb,str_patch);
        
        %%%%%%%%%%%%%%%%%%%%%
        %%% run overfeat 
        %%%%%%%%%%%%%%%%%%%%%
        cm1=['rm ',tmp_path,'*'];
        system(cm1);
        cm2=['cp ',str_patch,' ',tmp_path];
        system(cm2);
        cm3=[overfeat_path,' -i ',tmp_path,' -o ',feature_path];
        system(cm3);
        %%%%%%%%%%%%%%%%%%%%% 
        topo=[a.Area,a.MajorAxisLength,a.MinorAxisLength,a.Orientation];
        tarList{i}=struct('seg',bw,'id',[],'patch',maxID+numNewCell,'parent',[],...
            'child',[],'Centroid',a.Centroid,'props',topo);
               
        idxMap=zeros(1,numCell);
        idxMap(1)=i;
        for j=1:numCell-1
            
            bw=ismember(lab,j+1);
            cc=bwconncomp(bw);
            if(cc.NumObjects>1)
                bw=zeros(sz);
                numPixels = cellfun(@numel,cc.PixelIdxList);
                [~,biggest_idx] = max(numPixels);
                bw(cc.PixelIdxList{biggest_idx}) = 1;
            end
            im_region = tarImg;
            mask = imdilate(bw,se5);
            im_region(~mask)=0;
            a = regionprops(bw,'Centroid','Area','MajorAxisLength','MinorAxisLength','Orientation');
            x0=round(a.Centroid(2));y0=round(a.Centroid(1));
            
            %%% create image patch %%%
            tmp=zeros(patchSize,patchSize);
            h=min([x0-1,y0-1,20,dimx-x0,dimy-y0]);
            tmp(halfPatch-h:1:halfPatch+h, halfPatch-h:1:halfPatch+h)=...
                im_region(x0-h:1:x0+h,y0-h:1:y0+h);
            tmp=imresize(tmp,[overFeatSize,overFeatSize]);
            tmp=mat2gray(tmp);
            
            numNewCell=numNewCell+1;
            rgb=cat(3,tmp,tmp,tmp);
            str_patch=sprintf('%s/%03d.tif',str_seg_patch,maxID+numNewCell);
            imwrite(rgb,str_patch);
            
            %%%%%%%%%%%%%%%%%%%%%
            %%% run overfeat 
            %%%%%%%%%%%%%%%%%%%%%
            cm1=['rm ',tmp_path,'*'];
            system(cm1);
            cm2=['cp ',str_patch,' ',tmp_path];
            system(cm2);
            cm3=[overfeat_path,' -i ',tmp_path,' -o ',feature_path];
            system(cm3);
            %%%%%%%%%%%%%%%%%%%%%
            topo=[a.Area,a.MajorAxisLength,a.MinorAxisLength,a.Orientation];
            tmpCell=struct('seg',bw,'id',[],'patch',maxID+numNewCell,'parent',[],...
                'child',[],'Centroid',a.Centroid,'props',topo);
            
            srcList = cat(srcList,2,tmpCell);
            idxMap(j)=tarNum+numNewCell;
            clear tmpCell
        end
        
        %%%%%% assignment problem %%%%%%
        costMat=zeros(numCell,numCell);
        for j=1:1:numCell
            c1=srcList{pid(j)}.Centroid;
            for k=1:1:numCell
                c2=tarList{idxMap(k)}.Centroid;
                costMat(j,k)=norm(c1-c2);
            end
        end
        [assignment,~] = munkres(costMat);
        for j=1:1:numCell
            tmp=setdiff(srcList{pid(j)}.child,i); % in case of many-to-many matching
            srcList{pid(j)}.child=union(tmp,idxMap(assignment(j)));
            tarList{idxMap(assignment(j))}.parent=pid(j);
        end        
    end
end

%%%% update id %%%%%%
for i=1:1:tarNum
    pid=tarList{i}.parent;
    if(~isempty(pid))
        tarList{i}.id = srcList{pid}.id;
    else
        maxTrack=maxTrack+1;
        tarList{i}.id = maxTrack;
    end
end

