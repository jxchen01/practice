function nI=cutRegion(I,numCluster)

I=I>0;
%cmap=[1,0,0;0,0,1;0,1,0;0,1,1;1,1,0;1,0,1];

thresh=120;

I=bwareaopen(I,10);
I=imopen(I,strel('disk',2,0));
I=imfill(I,'holes');
[xdim,ydim]=size(I);

b=bwboundaries(I);

%%% draw the polygoal regions %%%
% poly=zeros(xdim,ydim);


tri=[];
vPoints=[];
GV=[];
offset=0;
for reg=1:1:numel(b)
    L=b{reg};
    vertexList=curveEvolution(L',size(L,1),0);
    L=vertexList';
    clear vertexList
    
%     %%%% add to poly %%%%
%     for i=2:1:size(L,1)
%         [xx,yy]=bresenham(L(i,1),L(i,2),L(i-1,1),L(i-1,2));
%         for j=1:numel(xx)
%             tx=round(xx(j)); ty=round(yy(j));
%             if(tx>xdim), tx = xdim; end
%             if(tx<1), tx=1; end
%             if(ty>ydim), ty = ydim; end
%             if(ty<1), ty=1; end
%             poly(tx,ty)=1;
%         end
%     end
    
    L(end,:)=[];
    
    np=size(L,1);
    C=zeros(np,2);
    C(:,1)=1:1:np;C(:,2)=2:1:np+1; C(end,2)=1;
    dt = delaunayTriangulation(L,C);
    clear C np L

    inside = isInterior(dt);
    pts=dt.ConnectivityList(inside,:);
    numVertex=size(pts,1);  %%%% number of vertex in the similaity graph
    G=zeros(numVertex,numVertex);

    tri=cat(1,tri,pts+offset); %%%% triangles
    vPoints=cat(1,vPoints,dt.Points); %%% vertices in the polygon
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%% concavity %%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    numPolygonVertice=size(dt.Points,1);
    concavePoint=zeros(numPolygonVertice,3);
    
    pv=zeros(3,2);
    pv(1,:)=dt.Points(end,:);
    pv(2:3,:)=dt.Points(1:2,:);
    pc=mean(pv,1);
    ti=pointLocation(dt,pc);
    if(~inside(ti))
        concavePoint(1,1)=1;
        v1=pv(1,:)-pv(2,:);
        v2=pv(3,:)-pv(2,:);
        concavePoint(1,2)=acosd(dot(v1,v2)/(norm(v1)*norm(v2)));
        vn=v1+v2;
        vn=vn./norm(vn);
        concavePoint(1,3)=vn(1); concavePoint(1,4)=vn(2);
    end
        
    for i=2:1:size(dt.Points,1)-1
        pv=dt.Points(i-1:i+1,:);
        pc=mean(pv,1);
        ti=pointLocation(dt,pc);
        if(~inside(ti))
            concavePoint(i,1)=1;
            v1=pv(1,:)-pv(2,:);
            v2=pv(3,:)-pv(2,:);
            concavePoint(i,2)=acosd(dot(v1,v2)/(norm(v1)*norm(v2)));
            vn=v1+v2;
            vn=vn./norm(vn);
            concavePoint(i,3)=vn(1); concavePoint(i,4)=vn(2); 
        end
    end
    
    pv=zeros(3,2);
    pv(3,:)=dt.Points(1,:);
    pv(1:2,:)=dt.Points(end-1:end,:);
    pc=mean(pv,1);
    ti=pointLocation(dt,pc);
    if(~inside(ti))
        concavePoint(end,1)=1;
        v1=pv(1,:)-pv(2,:);
        v2=pv(3,:)-pv(2,:);
        concavePoint(end,2)=acosd(dot(v1,v2)/(norm(v1)*norm(v2)));
        vn=v1+v2;
        vn=vn./norm(vn);
        concavePoint(end,3)=vn(1); concavePoint(end,4)=vn(2);
    end
    
    for i=1:1:numVertex
        pp=pts(i,:);
        if(concavePoint(pp(1),1) && concavePoint(pp(2),1) && concavePoint(pp(3),1))
            if(concavePoint(pp(1),2)<thresh && concavePoint(pp(2),2)<thresh && concavePoint(pp(3),2)<thresh)
                concavePoint(pp(1),1)=2;
                concavePoint(pp(2),1)=2;
                concavePoint(pp(3),1)=2;
            end
        end
    end
    clear pp
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %   triplot(pts,dt.Points(:,1),dt.Points(:,2));
 %   hold on
    for i=1:1:numVertex-1
        p1=pts(i,:);
        for j=i+1:1:numVertex
            p2=pts(j,:);
            if(numel(union(p1,p2))==4)
                G(i,j)=1;
                G(j,i)=1;
                
                %%%% modify the cost according to cancavity %%%%
                pn=intersect(p1,p2);
                
                if(concavePoint(pn(1),1)==2 && concavePoint(pn(2),1)==2)
                    G(i,j)=0.1;
                    G(j,i)=0.1;
                else
                
                    concaveCount=0;
                    for k=1:2
                        if(concavePoint(pn(k),1)>1e-5)
                            concaveCount = concaveCount + 1;
                        end
                    end
                    
                    vt=dt.Points(pn(2),:)-dt.Points(pn(1),:);
                    vt=vt./norm(vt);
                    for k=1:2
                        if(concavePoint(pn(k),1)>1e-5)
                            if(concavePoint(pn(k),2)<thresh)
                                if(k==2)
                                    vt=-vt;
                                end
                                if(dot(vt,concavePoint(pn(k),3:4))<-0.5)
                                    if(concaveCount==2)
                                        G(i,j)=G(i,j)*0.95; % extra contribution when double concave
                                    end
                                    G(i,j)=G(i,j)*max([60,concavePoint(pn(k),2)])/thresh;
                                    G(j,i)=G(i,j);
                                end
                            end
                        end
                    end
                    
                    if(concaveCount==2)
                        v1=concavePoint(pn(1),3:4);
                        v2=concavePoint(pn(2),3:4);
                        
                        t=norm(v1+v2)/2;
                        %t=abs(dot(v1,v2));
                        if(t<0.2-(1e-3) && concavePoint(pn(1),2)<thresh && concavePoint(pn(2),2)<thresh)
                            G(i,j)= max([0.1,(t+1e-3)]) * 5.0 * G(i,j);
                            G(j,i)=G(i,j);
                        end
                    end
                
                end
                
% %%%%%%%%%%%%% check the edge of each iteration %%%%%%%%%%%%%%%
%                 d1=dt.Points(pn(1),:);
%                 d2=dt.Points(pn(2),:);
%                 plot([d1(1),d2(1)],[d1(2),d2(2)],'-r');
%                 disp(G(i,j))
%                 keyboard
%                 triplot(pts,dt.Points(:,1),dt.Points(:,2));


%%%%%%%% draw a line connecting the centers of the triangles 
%%%%%%%% analyzed in this iteration 
%                 c1=dt.Points(pts(i,:),:);
%                 cp1=round(mean(c1,1));
%                 
%                 c2=dt.Points(pts(j,:),:);
%                 cp2=round(mean(c2,1));
%                 plot([cp1(1),cp2(1)],[cp1(2),cp2(2)],'r');     
            end
        end
    end
%    hold off
%     drawnow

    offset = offset + size(vPoints,1);

    GV = blkdiag(GV,G);
end


Cl = SpectralClustering(GV, numCluster, 3);
% t=ones(xdim,ydim);
% for i=1:1:size(tri,1)
%     p=vPoints(tri(i,:),:);
%     cp=round(mean(p,1));
%     t(cp(1),cp(2))=Cl(i)+1;
%     plot(cp(1),cp(2),'*','Color',cmap(Cl(i),:))
% end
% hold off 
% drawnow


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% build the new regions %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%poly=imfill(poly,'holes');

nI=zeros(xdim,ydim);
for i=1:1:size(tri,1)
    tmp=false(xdim,ydim);
    for j=1:1:2
        v1=vPoints(tri(i,j),:);
        for k=2:1:3
            v2=vPoints(tri(i,k),:);
            [xx,yy]=bresenham(v1(1),v1(2),v2(1),v2(2));
            for tt=1:1:numel(xx)
                tx=round(xx(tt)); ty=round(yy(tt));
                if(tx>xdim), tx = xdim; end
                if(tx<1), tx=1; end
                if(ty>ydim), ty = ydim; end
                if(ty<1), ty=1; end
                tmp(tx,ty)=true;
            end
        end
    end
    tmp=imfill(tmp,'holes');
    nI(tmp)=Cl(i);
end


% numTri=size(tri,1);
% for i=1:1:numTri-1
%     p1=tri(i,:);
%     for j=i+1:1:numTri
%         p2=tri(j,:);
%         pn=intersect(p1,p2);
%         if(numel(pn)==2)
%             if(Cl(i)~=Cl(j))
%                 %%% find a place to cut
%                 v1=vPoints(pn(1),:);
%                 v2=vPoints(pn(2),:);
%                 
%                 k=v1-v2; k=k./norm(k);
%                 for dt=1:1:20
%                     
%                 end
%             end
%         end
%     end
% end
