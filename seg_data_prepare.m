probPath='/home/jchen16/u-net-compiled/HeLa/02_RES/35000';
rescaleSize=0.5;
cellName='N2DL-HeLa';
dataset='train';
sq=2;
numFrame=92;
patchSize=101; halfPatch=(patchSize-1)/2+1;
bdThreshold=25;
overFeatSize=231;


se=strel('disk',5,0);
se3=strel('disk',3,0);

%%%%% get image size %%%%
str=sprintf('../data/%s/%s/%02d/t%02d.tif',cellName,dataset,sq,1);
I=mat2gray(imread(str));
[dimx,dimy]=size(I);

%%%%% create boundary mask %%%%%%
bdTemp=true(dimx,dimy);
bdTemp(1:bdThreshold,:)=false;
bdTemp(:,1:bdThreshold)=false;
bdTemp(end-bdThreshold+1:end,:)=false;
bdTemp(:,end-bdThreshold+1:end)=false;


%%%%% loop through each frame %%%%%%
for i=1:1:numFrame
    disp(i)
    
    SegPatchIdx=0;
    str2=sprintf('../data/%s/%s/%02d_SEG_PATCH/%02d',cellName,dataset,sq,i);
    if(~exist(str2,'dir'))
        mkdir(str2);
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
    bw=imfill(bw,'holes');

    %%% loop through each region %%% 
    cc=bwconncomp(bw);
    labmat = labelmatrix(cc);
    segFrame=cell(1,0);
    stat=regionprops(cc,'Centroid','Area','MajorAxisLength','MinorAxisLength','Orientation');
    numRegion = cc.NumObjects;
    for k=1:1:numRegion   
        sc=ismember(labmat,k);
        
        %%%% totally within the boundary mask %%%%
        checkTemp=sc & bdTemp;
        if(~any(checkTemp(:)))
            continue;
        end
        
        %%% update the segmentation information
        im_region = I_original;
        mask = imdilate(sc,se);
        im_region(~mask)=0;
        x0=round(stat(k).Centroid(2));y0=round(stat(k).Centroid(1));
        
%         %%% create image patch %%%
%         tmp=zeros(patchSize,patchSize);
%         h=min([x0-1,y0-1,20,dimx-x0,dimy-y0]);
%         tmp(halfPatch-h:1:halfPatch+h, halfPatch-h:1:halfPatch+h)=...
%             im_region(x0-h:1:x0+h,y0-h:1:y0+h);
%         tmp=imresize(tmp,[overFeatSize,overFeatSize]);
%         tmp=mat2gray(tmp);
%         
        SegPatchIdx=SegPatchIdx+1;
%         rgb=cat(3,tmp,tmp,tmp);
%         str=sprintf('%s/%03d.tif',str2,SegPatchIdx);
%         imwrite(rgb,str);
        
        topo=[stat(k).Area,stat(k).MajorAxisLength,stat(k).MinorAxisLength,stat(k).Orientation];
        
        tmpCell=struct('seg',sc,'id',[],'patch',SegPatchIdx,'parent',[],...
            'child',[],'Centroid',stat(k).Centroid,'props',topo);
        segFrame = cat(2,segFrame,tmpCell);
        clear tmpCell
    end
    
    str=sprintf('../data/%s/%s/%02d_SEG_DATA/data_%02d.mat',cellName,dataset,sq,i);
    save(str,'segFrame'); 
end
