probPath='/home/jchen16/u-net-compiled/HeLa/02_RES/35000';
rescaleSize=0.5;
cellName='N2DL-HeLa';
dataset='train';
sq=2;
numFrame=92;
patchSize=101;
halfPatch=(patchSize-1)/2+1;
bdThreshold=25;

overFeatSize=231;
se=strel('disk',5,0);
se3=strel('disk',3,0);

%%%%% get image size %%%%
str=sprintf('../data/%s/%s/%02d/t%02d.tif',cellName,dataset,sq,0);
I=imread(str);
[dimx,dimy]=size(I);

bdTemp=true(dimx,dimy);
bdTemp(1:bdThreshold,:)=false;
bdTemp(:,1:bdThreshold)=false;
bdTemp(end-bdThreshold+1:end,:)=false;
bdTemp(:,end-bdThreshold+1:end)=false;


%%%%% loop through each frame %%%%%
for i=1:1:numFrame
    disp(i)
    
    CellPatchIdx=0;
    str1=sprintf('../data/%s/%s/%02d_CELL_PATCH/%02d',cellName,dataset,sq,i);
    if(~exist(str1,'dir'))
        mkdir(str1);
    end
    
    SegPatchIdx=0;
    str2=sprintf('../data/%s/%s/%02d_SEG_PATCH/%02d',cellName,dataset,sq,i);
    if(~exist(str2,'dir'))
        mkdir(str2);
    end

    if(i>1)
        idMap0=idMap;
        cellFrame0=cellFrame;
        segFrame0=segFrame;
    end
    
    %%% load raw image %%%
    str=sprintf('../data/%s/%s/%02d/t%02d.tif',cellName,dataset,sq,i-1);
    I=mat2gray(imread(str));
    I_original = adapthisteq(I);
    I= imcomplement(I_original);

    %%% load segmentation result %%%

    str=sprintf('%s/prob_%d.tif',probPath,i);
    bw=mat2gray(imread(str));
    bw=im2bw(bw,graythresh(bw));
    bw=imresize(bw,rescaleSize);
    %str=sprintf('../data/%s/%s/%02d_SEG/%d.tif',cellName,dataset,sq,i);
    %bw=imread(str);
    %bw=bw>0;
    bw=imfill(bw,'holes');    


    %%% load tracking ground truth %%%
    str=sprintf('../data/%s/%s/%02d_GT/TRA/man_track%02d.tif',cellName,dataset,sq,i-1);
    track_lab_raw=imread(str);

    fn=track_lab_raw;
    fn(bw)=0;
    fn_candi=unique(nonzeros(fn(:)));

    track_lab_raw(~bw)=0;
    tp_idx=unique(nonzeros(track_lab_raw(:)));
    
    fn_idx=setdiff(fn_candi,tp_idx);

    track_lab=zeros(dimx,dimy);
    track_id = unique(nonzeros(track_lab_raw));
    for k=1:1:numel(track_id)
        tid=track_id(k);
        rgIdx=find(track_lab_raw==tid);
        track_lab(rgIdx(1))=tid;
    end
    clear rgIdx track_lab_raw track_id

    %%% loop through each region %%% 
    cc=bwconncomp(bw);
    stat=regionprops(cc,'Centroid','Area','MajorAxisLength','MinorAxisLength','Orientation');
    labmat = labelmatrix(cc);
    numRegion = cc.NumObjects;
    clear cc str
    
    segFrame=cell(1,0);
    cellFrame=cell(1,0);
    idMap=[];
    
    %%%%% false negative region first %%%%%
    for k=1:1:numel(fn_idx)
        idx=fn_idx(k);
        sc=ismember(fn,idx);
        a=regionprops(sc,'Centroid','Area','MajorAxisLength','MinorAxisLength','Orientation');
        
        im_region = I_original;
        mask = imdilate(sc,se);
        im_region(~mask)=0;
        x0=round(a.Centroid(2));y0=round(a.Centroid(1));
        tmp=zeros(patchSize,patchSize);
        h=min([x0-1,y0-1,20,dimx-x0,dimy-y0]);
        tmp(halfPatch-h:1:halfPatch+h, halfPatch-h:1:halfPatch+h)=...
            im_region(x0-h:1:x0+h,y0-h:1:y0+h);
        tmp=imresize(tmp,[overFeatSize,overFeatSize]);
        tmp=mat2gray(tmp);
        
        SegPatchIdx=SegPatchIdx+1;
        rgb=cat(3,tmp,tmp,tmp);
        str=sprintf('%s/%03d.tif',str2,SegPatchIdx);
        imwrite(rgb,str);
        
        clear im_region mask tmp h x0 y0
        
        topo=[a.Area,a.MajorAxisLength,a.MinorAxisLength,a.Orientation];
        
        tmpCell=struct('seg',sc,'id',idx,'patch',SegPatchIdx,'parent',[],...
            'child',[],'Centroid',a.Centroid,'props',topo);
        segFrame = cat(2,segFrame,tmpCell);
        clear tmpCell 
        
        CellPatchIdx = CellPatchIdx+1;
        str=sprintf('%s/%03d.tif',str1,CellPatchIdx);
        imwrite(rgb,str);
        
        tmpCell = struct('seg',sc,'id',idx,'patch',CellPatchIdx,'parent',[],...
            'child',[],'Centroid',a.Centroid,'props',topo);
        cellFrame=cat(2,cellFrame,tmpCell);
        idMap=cat(2,idMap,idx);
        
        clear tmpCell sc topo 
    end
    clear a 
    
    for k=1:1:numRegion     
        sc=ismember(labmat,k);
        idx=unique(nonzeros(track_lab(sc)));
        
        checkTemp=sc & bdTemp;
        if(numel(idx)==0 && ~any(checkTemp(:)))
            continue;
        end
        clear checkTemp
        
        %%% update the segmentation information
        im_region = I_original;
        mask = imdilate(sc,se);
        im_region(~mask)=0;
        x0=round(stat(k).Centroid(2));y0=round(stat(k).Centroid(1));
        tmp=zeros(patchSize,patchSize);
        h=min([x0-1,y0-1,20,dimx-x0,dimy-y0]);
        tmp(halfPatch-h:1:halfPatch+h, halfPatch-h:1:halfPatch+h)=...
            im_region(x0-h:1:x0+h,y0-h:1:y0+h);
        tmp=imresize(tmp,[overFeatSize,overFeatSize]);
        tmp=mat2gray(tmp);
        
        SegPatchIdx=SegPatchIdx+1;
        rgb=cat(3,tmp,tmp,tmp);
        str=sprintf('%s/%03d.tif',str2,SegPatchIdx);
        imwrite(rgb,str);
        
        clear im_region mask tmp h x0 y0
        
        topo=[stat(k).Area,stat(k).MajorAxisLength,stat(k).MinorAxisLength,stat(k).Orientation];
        
        tmpCell=struct('seg',sc,'id',idx,'patch',SegPatchIdx,'parent',[],...
            'child',[],'Centroid',stat(k).Centroid,'props',topo);
        segFrame = cat(2,segFrame,tmpCell);
        clear tmpCell 
        
        %%% update the ground truth information
        if(numel(idx)==1)
            %%%%% true positive %%%% 
            CellPatchIdx = CellPatchIdx+1;
            str=sprintf('%s/%03d.tif',str1,CellPatchIdx);
            imwrite(rgb,str);
            
            tmpCell = struct('seg',sc,'id',idx,'patch',CellPatchIdx,'parent',[],...
                'child',[],'Centroid',stat(k).Centroid,'props',topo);
            cellFrame=cat(2,cellFrame,tmpCell);
            idMap=cat(2,idMap,idx);
            
            clear tmpCell sc topo str
        elseif(numel(idx)>1)
            %%% need to cut %%%
            % get the seeds (marker in the ground truth)
            seeds=ismember(track_lab,idx);
            
            % get the binary mask
            sc_mask=sc | xor(imdilate(seeds,se),seeds);
            sc_mask(1,:)=0; sc_mask(end,:)=0; sc_mask(:,1)=0; sc_mask(:,end)=0;
            sc_mask=imfill(sc_mask,'holes');
            
            % distance transformation 
            distmap=bwdist(seeds);
            distmap(~sc_mask)=0; 
            
            % combine distance cue and image cue
            aa=0.5.*mat2gray(distmap)+ mat2gray(I);
            aa=aa./1.5;
            
            % perform marker controlled watershed
            I_mod =imimposemin(aa, ~sc_mask|seeds);
            L = watershed(I_mod);
            
            clear sc_mask I_mod aa distmap
            
            if(max(L(:))~=numel(idx)+1) % watershed didn't get the correct number of regions
                
                % cut by voronoi diagram (outer-centerline)
                b=imcomplement(seeds);
                bctl=bwmorph(b,'thin',Inf);
                L=L>1;
                L(bctl>0)=0;
                clear bctl b
                
                % check each split region
                numNew=0;
                new_cc = bwconncomp(L,4);
                a=regionprops(new_cc,'Centroid','Area','MajorAxisLength','MinorAxisLength','Orientation');
                lab_cut=labelmatrix(new_cc);
                for j=1:1:new_cc.NumObjects
                    scs=ismember(lab_cut,j);
                    nidx=unique(nonzeros(track_lab(scs)));
                    if(numel(nidx)==1)    
                        numNew = numNew + 1;
                        
                        im_region = I_original;
                        mask = imdilate(scs,se);
                        im_region(~mask)=0;
                        x0=round(a(j).Centroid(2));y0=round(a(j).Centroid(1));
                        tmp=zeros(patchSize,patchSize);
                        h=min([x0-1,y0-1,20,dimx-x0,dimy-y0]);
                        tmp(halfPatch-h:1:halfPatch+h, halfPatch-h:1:halfPatch+h)=...
                            im_region(x0-h:1:x0+h,y0-h:1:y0+h);
                        tmp=imresize(tmp,[overFeatSize,overFeatSize]);
                        tmp=mat2gray(tmp);
                        
                        CellPatchIdx=CellPatchIdx+1;
                        rgb=cat(3,tmp,tmp,tmp);
                        str=sprintf('%s/%03d.tif',str1,CellPatchIdx);
                        imwrite(rgb,str);
                        
                        topo=[a(j).Area,a(j).MajorAxisLength,a(j).MinorAxisLength,a(j).Orientation];
        
                        tmpCell=struct('seg',scs,'id',nidx,'patch',CellPatchIdx,...
                            'parent',[],'child',[],'Centroid',a(j).Centroid,'props',topo);
                        cellFrame=cat(2,cellFrame,tmpCell);
                        idMap=cat(2,idMap,nidx);
                        
                        clear tmpCell tmp h x0 y0 im_region mask topo
                    end
                end
                clear a
                
                if(numNew~=numel(idx))
                    disp('error in separating cells');
                    keyboard;
                end

                clear lab_cut new_cc seeds b bctl
            else
                % watershed works well
                L=L>1;
                
                % check each split region
                numNew=0;
                new_cc = bwconncomp(L,4);
                a=regionprops(new_cc,'Centroid','Area','MajorAxisLength','MinorAxisLength','Orientation');
                lab_cut=labelmatrix(new_cc);
                for j=1:1:new_cc.NumObjects
                    scs=ismember(lab_cut,j);
                    nidx=unique(nonzeros(track_lab(scs)));
                    if(numel(nidx)==1)    
                        numNew = numNew + 1;
                        
                        im_region = I_original;
                        mask = imdilate(scs,se);
                        im_region(~mask)=0;
                        x0=round(a(j).Centroid(2));y0=round(a(j).Centroid(1));
                        tmp=zeros(patchSize,patchSize);
                        h=min([x0-1,y0-1,20,dimx-x0,dimy-y0]);
                        tmp(halfPatch-h:1:halfPatch+h, halfPatch-h:1:halfPatch+h)=...
                            im_region(x0-h:1:x0+h,y0-h:1:y0+h);
                        tmp=imresize(tmp,[overFeatSize,overFeatSize]);
                        tmp=mat2gray(tmp);

                        CellPatchIdx = CellPatchIdx +1;
                        rgb=cat(3,tmp,tmp,tmp);
                        str=sprintf('%s/%03d.tif',str1,CellPatchIdx);
                        imwrite(rgb,str);
                        
                        topo=[a(j).Area,a(j).MajorAxisLength,a(j).MinorAxisLength,a(j).Orientation];
                        
                        tmpCell=struct('seg',scs,'id',nidx,'patch',CellPatchIdx,...
                            'parent',[],'child',[],'Centroid',a(j).Centroid,'props',topo);
                        cellFrame=cat(2,cellFrame,tmpCell);
                        idMap=cat(2,idMap,nidx);
                        
                        clear tmpCell tmp h x0 y0 im_region mask topo
                    end
                end
                clear a
            end
            clear I_mod L scs nidx 
        end
    end
  
    if(i>1)
        for j=1:1:numel(cellFrame)
            cid=cellFrame{j}.id;
            pidx=find(idMap0==cid);
            if(numel(pidx)==1)
                cellFrame{j}.parent=pidx;
                cellFrame0{pidx}.child=cat(2,cellFrame0{pidx}.child,j);
            elseif(numel(pidx)>1)
                disp('error in finding parent');
                keyboard;
            end
        end
        str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,i-1);
        save(str,'cellFrame0','segFrame0');
        
        if(i==numFrame)
            cellFrame0=cellFrame; 
            segFrame0=segFrame;
            str=sprintf('../data/%s/%s/%02d_CELL/data_%02d.mat',cellName,dataset,sq,i);
            save(str,'cellFrame0','segFrame0');
        end
    end
    
    clear bw  track_lab I_original I labmat

end

