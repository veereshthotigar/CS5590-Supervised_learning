inputFolder = '\\kc.umkc.edu\kc-users\home\v\vmt9mm\Desktop\LDA_face_recognition\att_faces';
X = [];
Y = [];
for i = 1 : 40
	% Get a list of all files in the folder with the desired file name pattern.
	insideFolder = strcat(inputFolder,'\s');
	for k = 1 : 5
        %baseFileName = theFiles(k).name;
        fullFileName = strcat(inputFolder,strcat(strcat('/s',num2str(i,'%d')),strcat('/',strcat(num2str(k,'%d'),'.pgm'))));
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
        fullFileName = strcat(inputFolder,strcat(strcat('/s',num2str(i,'%d')),strcat('/',strcat(num2str(k,'%d'),'.pgm'))));
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

[r,c] = size(train_data);
% Compute the mean of the data matrix "The mean of each row"
m = mean(train_data')';
% Subtract the mean from each image [Centering the data]
d=double(train_data)-repmat(m,1,c);


% Compute the covariance matrix (co)
co=d*d';

% Compute the eigen values and eigen vectors of the covariance matrix
[eigvector,eigvl]=eig(co);


% Sort the eigen vectors according to the eigen values
eigvalue = diag(eigvl);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

% Compute the number of eigen values that greater than zero (you can select any threshold)
count1=0;
for i=1:size(eigvalue,1)
    if(eigvalue(i)>0)
        count1=count1+1;
    end
end

% We can use all the eigen vectors but this method will increase the
% computation time and complixity
%vec=eigvector(:,:);

% And also we can use the eigen vectors that the corresponding eigen values is greater than zero(Threshold) and this method will decrease the
% computation time and complixity

% Creating PCA Subspace
vec=eigvector(:,1:count1);

% Projecting training data into PCA subspace
x=vec'*d;  

% Projecting test data into PCA subspace
t=Y;
%Subtract the mean from the test data
t=double(t)-m;
t=vec'*t;

diff=pdist2(x',t'); %% Finding Eucledian Distances betweern Train and Test%%
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

save('PCAnormalizedscore.mat','normmat');
% Evaluation of Metric GAR, FAR and plotting ROC 
%[roc,EER,area,EERthr,ALLthr,d,gen,imp] = ezroc13(Predicted_respone,true_response,2,'',1);

