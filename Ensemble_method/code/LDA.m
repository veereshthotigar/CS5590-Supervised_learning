inputPath = '\\kc.umkc.edu\kc-users\home\v\vmt9mm\Desktop\LDA_face_recognition\att_faces';
X = [];
Y = [];
for i = 1 : 40
	% Get a list of all files in the folder with the desired file name pattern.
	insideFolder = strcat(inputPath,'\s');

	for k = 1 : 5
        %baseFileName = theFiles(k).name;
        fullFileName = strcat(inputPath,strcat(strcat('/s',num2str(i,'%d')),strcat('/',strcat(num2str(k,'%d'),'.pgm'))));
        fprintf(1, 'Now reading %s\n', fullFileName);
          % such as reading it in as an image array with imread()
        img = imread(fullFileName);
        [r,c] = size(img);
        temp = reshape(img',r*c,1);  %% Reshaping 2D images into 1D image vectors
                               %% here img' is used because reshape(A,M,N) function reads the matrix A columnwise
                               %% where as an image matrix is constructed with first N pixels as first row,next N in second row so on
        X = [X temp];                %% X,the image matrix with columnsgetting added for each image
    end
    for k = 6 : 10
        %baseFileName = theFiles(k).name;
        fullFileName = strcat(inputPath,strcat(strcat('/s',num2str(i,'%d')),strcat('/',strcat(num2str(k,'%d'),'.pgm'))));
          % Now do whatever you want with this file name,
          % such as reading it in as an image array with imread()
        img = imread(fullFileName);
        [r,c] = size(img);
        temp = reshape(img',r*c,1);  %% Reshaping 2D images into 1D image vectors
                               %% here img' is used because reshape(A,M,N) function reads the matrix A columnwise
                               %% where as an image matrix is constructed with first N pixels as first row,next N in second row so on
        Y = [Y temp];                %% X,the image matrix with columnsgetting added for each image
    end
end

train_data = X;
test_data = Y;

d=double(train_data);
m=mean(d,2);
M=repmat(m,[1,200]); %% Calculating Mean of data set%%

j=1;
for i=0:5:195
mea(:,j)=mean(d(:,i+1:i+5),2);
me(:,i+1:i+5)=repmat(mea(:,j),[1,5]); %% Calculating Mean of Each Class%%
j=j+1;
end

temp=zeros(10304,10304);
wsca=zeros(10304,10304);
for i =0:5:195
temp=(d(:,i+1:i+5)-me(:,i+1:i+5))*((d(:,i+1:i+5)-me(:,i+1:i+5))');
wsca=temp+wsca; %%calculating with in scatter matrix%%
end
temp1=zeros(10304,10304);
bsca=zeros(10304,10304);

for i=1:40
temp1=(mea(:,i)-m)*((mea(:,i)-m)');
bsca=temp1+bsca; %%Calculating between scatter matrix%%
end

Va=cov(((d-M)'));
[PCAV,PCAD]=eig(Va,'vector');
PCAVk=PCAV(:,(10304-159:10304)); %% PCA Eigen Space selection%%
wscaproj=PCAVk'*wsca*PCAVk;
bscaproj=PCAVk'*bsca*PCAVk; %% within scatter and between projecting into PCA
[V,D]=eig(bscaproj,wscaproj,'vector');
Proj=PCAVk*V;
Projk=Proj(:,1:39); %% LDA Eigen Space selection%%
trainpro=Projk'*(d-M); %% Train Projection onto Eigen Space%%

y1=test_data;
testD=double(y1);
testm=mean(testD,2);
testM=repmat(testm,[1,200]);
testpro=Projk'*(testD-testM);
diff=pdist2(trainpro',testpro'); %% Finding Eucledian Distances betweern Train and Test%%
norm=max(diff(:));
normmat=1/norm*(diff);

for i=1:200
for j=1:200
if(normmat(i,j)>0.50)
normmat(i,j)=1; %% Setting Threshold%%
else
normmat(i,j)=0;
end
end
end

tar=[zeros(1,5),ones(1,195)];
tar=[tar;tar;tar;tar;tar];
target=zeros(200,200);
target(1:5,:)=tar;
for i=5:5:195
target(i+1:i+5,:)=circshift(tar,i,2); %% Creating Targets for each subject%%
end
save('LDAnormalizedscore.mat','normmat')
%{
ezroc3(normmat,target,2,' ',1); %% ROC Plot%%

temporary=normmat;
TP=zeros(200,1);
FN=zeros(200,1);
for i=1:200
for j=1:5
if (temporary(i,j)==0)
TP(i,1)=TP(i,1)+1;
else %% Finding True Positives and False Negatives
FN(i,1)=FN(i,1)+1;
end
end
if(rem(i,5)==0)
temporary=circshift(temporary,-5,2);
end
end

temporary=normmat;
GAR=TP/5;
FRR=FN/5;
TN=zeros(200,1);
FP=zeros(200,1);
for i=1:200
for j=6:200
if(temporary(i,j)==1)
TN(i,1)=TN(i,1)+1;
else
FP(i,1)=FP(i,1)+1; %% Calculating False Positives and True Negatives%%
end
if(rem(i,5)==0)
temporary=circshift(temporary,-5,2);
end
end
end

GRR=TN/195;
FAR=FP/195;
mean_GAR=mean(GAR);
mean_GRR=mean(GRR);
mean_FAR=mean(FAR);
mean_FRR=mean(FRR);
%}