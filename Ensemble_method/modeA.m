%import PCA and LDA matrices
PCA = importdata('PCAscores.mat');
LDA = importdata('LDAscores.mat');
%initialize matrix for min,max, avg
mainMin = [];
mainMax = [];
mainAvg = [];


%Create the label matrix
zeroMatrix = zeros(5);
oneMatrix = ones(5);
labelsMain = [];
labelCount = 1;
generalMat = [];
for i=1:40 
    generalMat = [];
    for j=1:40
        if(labelCount==j)
            generalMat = horzcat(generalMat,zeroMatrix);
        else
            generalMat = horzcat(generalMat,oneMatrix);
        end
    end
    labelsMain = vertcat(labelsMain,generalMat);
    labelCount = labelCount + 1;
end

%Create max matrix
for c = 1:200
    for r = 1:200
        mainMax(c,r)=max(PCA(c,r),LDA(c,r));
    end
end

%Create min matrix
for c = 1:200
    for r = 1:200
        mainMin(c,r)=min(PCA(c,r),LDA(c,r));
    end
end

%Create avg matrix
for c = 1:200
    for r = 1:200 
        mainAvg(c,r)=((PCA(c,r)+LDA(c,r))/2);
    end
end



%Utilize ezroc function to evaluate performance for min,max,avg,LDA,PCA
avgPlot =  ezroc3(mainAvg,labelsMain,2,'',1);
minPlot =  ezroc3(mainMin,labelsMain,2,'',1);
maxPlot =  ezroc3(mainMax,labelsMain,2,'',1);
LDAPlot =  ezroc3(LDA,labelsMain,2,'',1);
PCAPlot =  ezroc3(PCA,labelsMain,2,'',1);

%Obtain x and y for avg
avgX = [];
avgY = [];
for c = 1:503
    avgX(1,c)=avgPlot(1,c);
    avgY(2,c)=avgPlot(2,c);
end

%Obtain x and y for min
minX = []
minY = []
for c = 1:503
    minX(1,c)=minPlot(1,c);
    minY(2,c)=minPlot(2,c);
end

%Obtain x and y for max
maxX = []
maxY = []
for c = 1:503
    maxX(1,c)=maxPlot(1,c);
    maxY(2,c)=maxPlot(2,c);
end

%Obtain x and y for LDA plot
ldaX = []
ldaY = []
for c = 1:503
    ldaX(1,c)=LDAPlot(1,c);
    ldaY(2,c)=LDAPlot(2,c);
end

%Obtain x and y for PCA plot
PCAX = [];
PCAY = [];
for c = 1:503
    PCAX(1,c)=PCAPlot(1,c);
    PCAY(2,c)=PCAPlot(2,c);
end

%Create comparison plot
plot(ldaY,ldaX,'color','y')
title('LDA')
hold on
plot(avgY,avgX,'color','b')
title('Avg')
hold on
plot(minY,minX,'color','r')
title('Min')
hold on
plot(maxY,maxX,'color','g')
title('Max')
hold on
plot(PCAY,PCAX,'color','c')
hold off

%Add legend and features to plot
legend('\color{yellow} LDA','\color{blue} Average','\color{red} Minimum','\color{green} Max', '\color{cyan} PCA')
title('MCS Score Level')
xlabel('FAR')
ylabel('GAR')