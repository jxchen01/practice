function vertexList=curveEvolution(plist,initNum,drawFlag)
%%%% ASSUMPTION (only in step 3): closed curve
% curve evolution to simpliy the polygonal curve%
if(size(plist,1)~=2)
    if(size(plist,2)~=2)
        error('dimension error');
    else
        plist=plist';
    end
end

%%%%%%% parameters %%%%%%%%%
minVerNum=40;
minType1=60;
minType2=15;
minVerDist=2;

% initialization: simplify to polygonal curves
% (step 0: pre-prune, only keep non-trivial points)
dx=plist(1,2)-plist(1,1);
dy=plist(2,2)-plist(2,1);
toRm=zeros(1,initNum);
for i=2:1:initNum-1
    cdx=plist(1,i+1)-plist(1,i);
    cdy=plist(2,i+1)-plist(2,i);
    if(cdx == dx && cdy==dy)
        toRm(1,i)=1;
    else
        dx=cdx; dy=cdy;
    end
end

idx=find(toRm==0);
numVertex=numel(idx);
vertexList=plist(:,idx);
clear idx toRm dx dy cdx cdy plist initNum

% step 1: iteratively remove vertices (type 1)
relevanceMeasure=zeros(1,numVertex);
relevanceMeasure(1,1)=100000;
relevanceMeasure(1,end)=100000;
x0=vertexList(1,1);y0=vertexList(2,1);
x2=vertexList(1,2);y2=vertexList(2,2);
for i=2:1:(numVertex-1)
    x1=x0;y1=y0;
    x0=x2;y0=y2;
    x2=vertexList(1,i+1);
    y2=vertexList(2,i+1);
    d1=[x1-x0,y1-y0]; L1=norm(d1);
    d2=[x2-x0,y2-y0]; L2=norm(d2);
    relevanceMeasure(1,i) = (180-acosd(dot(d1,d2)/(L1*L2)))*L1*L2/(L1+L2);   
end

TotalVertexOld=numVertex;
while(1)
    [v,idx]=min(relevanceMeasure);
    if(v>minType1 || numVertex<minVerNum)
        break;
    end
    
    tmpIdx=(1:1:numVertex)~=idx;
    vertexList=vertexList(:,tmpIdx);
    relevanceMeasure=relevanceMeasure(1,tmpIdx);

    % compute relavance measure
    if(idx<numVertex-1)
        x0=vertexList(1,idx);y0=vertexList(2,idx);
        x1=vertexList(1,idx-1);y1=vertexList(2,idx-1);
        x2=vertexList(1,idx+1);y2=vertexList(2,idx+1);
        d1=[x1-x0,y1-y0]; L1=norm(d1);
        d2=[x2-x0,y2-y0]; L2=norm(d2);
        relevanceMeasure(1,idx) = (180-acosd(dot(d1,d2)/(L1*L2)))*L1*L2/(L1+L2);
    end
    
    if(idx>2)
        x2=vertexList(1,idx);y2=vertexList(2,idx);
        x0=vertexList(1,idx-1);y0=vertexList(2,idx-1);
        x1=vertexList(1,idx-2);y1=vertexList(2,idx-2);
        d1=[x1-x0,y1-y0]; L1=norm(d1);
        d2=[x2-x0,y2-y0]; L2=norm(d2);
        relevanceMeasure(1,idx-1) = (180-acosd(dot(d1,d2)/(L1*L2)))*L1*L2/(L1+L2);
    end
    
    numVertex=numVertex-1;
    
    if(drawFlag>0)
        xx=vertexList(1,:);
        yy=vertexList(2,:);
        plot(xx,yy,'-o');
        title(['interation : ', num2str(TotalVertexOld-numVertex)])
        drawnow;
        %pause(0.1)
    end
end

% step 2: remove flat vertex, maybe with long edges (type 2)
relevanceMeasure=zeros(1,numVertex);
relevanceMeasure(1,1)=180;
relevanceMeasure(1,end)=180;
x0=vertexList(1,1);y0=vertexList(2,1);
x2=vertexList(1,2);y2=vertexList(2,2);
for i=2:1:(numVertex-1)
    x1=x0;y1=y0;
    x0=x2;y0=y2;
    x2=vertexList(1,i+1);
    y2=vertexList(2,i+1);
    d1=[x1-x0,y1-y0]; L1=norm(d1);
    d2=[x2-x0,y2-y0]; L2=norm(d2);
    relevanceMeasure(1,i) = 180-acosd(dot(d1,d2)/(L1*L2));
end

while(1)
    [v,idx]=min(relevanceMeasure);
    if(v>minType2 || numVertex<20)
        break;
    end
    
    tmpIdx=(1:1:numVertex)~=idx;
    vertexList=vertexList(:,tmpIdx);
    relevanceMeasure=relevanceMeasure(1,tmpIdx);

    % compute relavance measure
    if(idx<numVertex-1)
        x0=vertexList(1,idx);y0=vertexList(2,idx);
        x1=vertexList(1,idx-1);y1=vertexList(2,idx-1);
        x2=vertexList(1,idx+1);y2=vertexList(2,idx+1);
        d1=[x1-x0,y1-y0]; L1=norm(d1);
        d2=[x2-x0,y2-y0]; L2=norm(d2);
        relevanceMeasure(1,idx) = 180-acosd(dot(d1,d2)/(L1*L2));
    end
    
    if(idx>2)
        x2=vertexList(1,idx);y2=vertexList(2,idx);
        x0=vertexList(1,idx-1);y0=vertexList(2,idx-1);
        x1=vertexList(1,idx-2);y1=vertexList(2,idx-2);
        d1=[x1-x0,y1-y0]; L1=norm(d1);
        d2=[x2-x0,y2-y0]; L2=norm(d2);
        relevanceMeasure(1,idx-1) = 180-acosd(dot(d1,d2)/(L1*L2));
    end
    
    numVertex=numVertex-1;
    
    if(drawFlag>0)
        xx=vertexList(1,:);
        yy=vertexList(2,:);
        plot(xx,yy,'-o');
        title(['interation : ', num2str(TotalVertexOld-numVertex)])
        drawnow;
        %pause(0.1)
    end
end

%%%% step 3: remove extremely short edges %%%%%%
relevanceMeasure=-1.*ones(1,numVertex);
for i=1:1:numVertex
    x1=vertexList(1,i);
    y1=vertexList(2,i);
    if(i==numVertex)
        nid=1;
    else
        nid=i+1;
    end
    x2=vertexList(1,nid);
    y2=vertexList(2,nid);
    
    a=norm([x1-x2,y1-y2]);
    if(a<minVerDist)
        if(i==1)
            x0=vertexList(1,end);
            y0=vertexList(2,end);
        else
            x0=vertexList(1,i-1);
            y0=vertexList(2,i-1);
        end
        b=norm([x0-x1,y0-y1]);
        c=norm([x0-x2,y0-y2]);
        t1=0.5*(b^2+c^2-a^2)/(b*c);
        if(i==numVertex)
            x3=vertexList(1,2);
            y3=vertexList(2,2);
        elseif(i==numVertex-1)
            x3=vertexList(1,1);
            y3=vertexList(2,1);
        else
            x3=vertexList(1,i+2);
            y3=vertexList(2,i+2);
        end
        d=norm([x3-x1,y3-y1]);
        e=norm([x3-x2,y3-y2]);
        t2=0.5*(d^2+e^2-a^2)/(d*e);
        if(t1>t2 && t1>relevanceMeasure(1,i))
            relevanceMeasure(1,i)=t1;
        elseif(t2>t1)
            relevanceMeasure(1,nid)=t2;
        end
    end
end

while(1)
    [v,idx]=max(relevanceMeasure);
    if(v<-0.9 || numVertex<20)
        break;
    end
    
    tmpIdx=(1:1:numVertex)~=idx;
    vertexList=vertexList(:,tmpIdx);
    numVertex=numVertex-1;
    relevanceMeasure=-1.*ones(1,numVertex);

    % compute relavance measure
    for i=1:1:numVertex
        x1=vertexList(1,i);
        y1=vertexList(2,i);
        if(i==numVertex)
            nid=1;
        else
            nid=i+1;
        end
        x2=vertexList(1,nid);
        y2=vertexList(2,nid);
        
        a=norm([x1-x2,y1-y2]);
        if(a<minVerDist)
            if(i==1)
                x0=vertexList(1,end);
                y0=vertexList(2,end);
            else
                x0=vertexList(1,i-1);
                y0=vertexList(2,i-1);
            end
            b=norm([x0-x1,y0-y1]);
            c=norm([x0-x2,y0-y2]);
            t1=0.5*(b^2+c^2-a^2)/(b*c);
            if(i==numVertex)
                x3=vertexList(1,2);
                y3=vertexList(2,2);
            elseif(i==numVertex-1)
                x3=vertexList(1,1);
                y3=vertexList(2,1);
            else
                x3=vertexList(1,i+2);
                y3=vertexList(2,i+2);
            end
            d=norm([x3-x1,y3-y1]);
            e=norm([x3-x2,y3-y2]);
            t2=0.5*(d^2+e^2-a^2)/(d*e);
            if(t1>t2 && t1>relevanceMeasure(1,i))
                relevanceMeasure(1,i)=t1;
            elseif(t2>t1)
                relevanceMeasure(1,nid)=t2;
            end
        end
    end
    
    if(drawFlag>0)
        xx=vertexList(1,:); xx(:,numVertex+1) = xx(:,1);
        yy=vertexList(2,:); yy(:,numVertex+1) = yy(:,1);
        plot(xx,yy,'-o');
        title(['interation : ', num2str(TotalVertexOld-numVertex)])
        drawnow;
    end
end


% step 4: step 1 again
relevanceMeasure=zeros(1,numVertex);
relevanceMeasure(1,1)=100000;
relevanceMeasure(1,end)=100000;
x0=vertexList(1,1);y0=vertexList(2,1);
x2=vertexList(1,2);y2=vertexList(2,2);
for i=2:1:(numVertex-1)
    x1=x0;y1=y0;
    x0=x2;y0=y2;
    x2=vertexList(1,i+1);
    y2=vertexList(2,i+1);
    d1=[x1-x0,y1-y0]; L1=norm(d1);
    d2=[x2-x0,y2-y0]; L2=norm(d2);
    relevanceMeasure(1,i) = (180-acosd(dot(d1,d2)/(L1*L2)))*L1*L2/(L1+L2);   
end

TotalVertexOld=numVertex;
while(1)
    [v,idx]=min(relevanceMeasure);
    if(v>minType1 || numVertex<minVerNum)
        break;
    end
    
    tmpIdx=(1:1:numVertex)~=idx;
    vertexList=vertexList(:,tmpIdx);
    relevanceMeasure=relevanceMeasure(1,tmpIdx);

    % compute relavance measure
    if(idx<numVertex-1)
        x0=vertexList(1,idx);y0=vertexList(2,idx);
        x1=vertexList(1,idx-1);y1=vertexList(2,idx-1);
        x2=vertexList(1,idx+1);y2=vertexList(2,idx+1);
        d1=[x1-x0,y1-y0]; L1=norm(d1);
        d2=[x2-x0,y2-y0]; L2=norm(d2);
        relevanceMeasure(1,idx) = (180-acosd(dot(d1,d2)/(L1*L2)))*L1*L2/(L1+L2);
    end
    
    if(idx>2)
        x2=vertexList(1,idx);y2=vertexList(2,idx);
        x0=vertexList(1,idx-1);y0=vertexList(2,idx-1);
        x1=vertexList(1,idx-2);y1=vertexList(2,idx-2);
        d1=[x1-x0,y1-y0]; L1=norm(d1);
        d2=[x2-x0,y2-y0]; L2=norm(d2);
        relevanceMeasure(1,idx-1) = (180-acosd(dot(d1,d2)/(L1*L2)))*L1*L2/(L1+L2);
    end
    
    numVertex=numVertex-1;
    
    if(drawFlag>0)
        xx=vertexList(1,:);
        yy=vertexList(2,:);
        plot(xx,yy,'-o');
        title(['interation : ', num2str(TotalVertexOld-numVertex)])
        drawnow;
        %pause(0.1)
    end
end

% step 5: step 2 again
relevanceMeasure=zeros(1,numVertex);
relevanceMeasure(1,1)=180;
relevanceMeasure(1,end)=180;
x0=vertexList(1,1);y0=vertexList(2,1);
x2=vertexList(1,2);y2=vertexList(2,2);
for i=2:1:(numVertex-1)
    x1=x0;y1=y0;
    x0=x2;y0=y2;
    x2=vertexList(1,i+1);
    y2=vertexList(2,i+1);
    d1=[x1-x0,y1-y0]; L1=norm(d1);
    d2=[x2-x0,y2-y0]; L2=norm(d2);
    relevanceMeasure(1,i) = 180-acosd(dot(d1,d2)/(L1*L2));
end

while(1)
    [v,idx]=min(relevanceMeasure);
    if(v>minType2 || numVertex<20)
        break;
    end
    
    tmpIdx=(1:1:numVertex)~=idx;
    vertexList=vertexList(:,tmpIdx);
    relevanceMeasure=relevanceMeasure(1,tmpIdx);

    % compute relavance measure
    if(idx<numVertex-1)
        x0=vertexList(1,idx);y0=vertexList(2,idx);
        x1=vertexList(1,idx-1);y1=vertexList(2,idx-1);
        x2=vertexList(1,idx+1);y2=vertexList(2,idx+1);
        d1=[x1-x0,y1-y0]; L1=norm(d1);
        d2=[x2-x0,y2-y0]; L2=norm(d2);
        relevanceMeasure(1,idx) = 180-acosd(dot(d1,d2)/(L1*L2));
    end
    
    if(idx>2)
        x2=vertexList(1,idx);y2=vertexList(2,idx);
        x0=vertexList(1,idx-1);y0=vertexList(2,idx-1);
        x1=vertexList(1,idx-2);y1=vertexList(2,idx-2);
        d1=[x1-x0,y1-y0]; L1=norm(d1);
        d2=[x2-x0,y2-y0]; L2=norm(d2);
        relevanceMeasure(1,idx-1) = 180-acosd(dot(d1,d2)/(L1*L2));
    end
    
    numVertex=numVertex-1;
    
    if(drawFlag>0)
        xx=vertexList(1,:);
        yy=vertexList(2,:);
        plot(xx,yy,'-o');
        title(['interation : ', num2str(TotalVertexOld-numVertex)])
        drawnow;
        %pause(0.1)
    end
end

% disp(TotalVertexOld)
% disp(numVertex)