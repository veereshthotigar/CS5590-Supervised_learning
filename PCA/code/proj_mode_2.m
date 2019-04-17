clear all
clear memory
clc
% Specify the folder where the files live.
inputFolder = 'C:\Users\VThotigar\Downloads\PCA_Project\Data';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(inputFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', inputFolder);
  uiwait(warndlg(errorMessage));
  return;
end
modeldata = [];
for i = 1 : 25
	% Get a list of all files in the folder with the desired file name pattern.
	insideFolder = strcat(inputFolder,'\s');
	SubFolder{i} = [insideFolder '' num2str(i,'%d')];
	filePattern = fullfile(SubFolder{i}, '*.pgm'); 
	theFiles = dir(filePattern);
	for k = 1 : length(theFiles)
	  baseFileName = theFiles(k).name;
	  fullFileName = fullfile(SubFolder{i}, baseFileName);
	  % Now do whatever you want with this file name,
	  % such as reading it in as an image array with imread()
	  imageArray = imread(fullFileName);
      [r0,c0] = size(imageArray);
      temp = reshape(imageArray',r0*c0,1);
      modeldata = [modeldata temp];
	  %imshow(imageArray);  % Display image.
	  %drawnow; % Force display to update immediately.
	end
end
traindata = [];
testdata = [];
for i = 26 : 40
	% Get a list of all files in the folder with the desired file name pattern.
	insideFolder = strcat(inputFolder,'\s');
	SubFolder{i} = [insideFolder '' num2str(i,'%d')];
	filePattern = fullfile(SubFolder{i}, '*.pgm'); 
	theFiles = dir(filePattern);
	 % train data
	for k = 1 : length(theFiles)/2
	  baseFileName = theFiles(k).name;
	  fullFileName = fullfile(SubFolder{i}, baseFileName);
	  % read file
	  ia1 = imread(fullFileName);
      [r1,c1] = size(ia1);
      t1 = reshape(ia1',r1*c1,1);
      traindata = [traindata t1];
    end
    % test data
    for k = length(theFiles)/2+1 : length(theFiles)
	  baseFileName = theFiles(k).name;
	  fullFileName = fullfile(SubFolder{i}, baseFileName);
	  % read file
	  ia2 = imread(fullFileName);
      [r2,c2] = size(ia2);
      t2 = reshape(ia2',r2*c2,1);
      testdata = [testdata t2];
	end
end

[tr,tc] = size(modeldata);
modeldata = double(modeldata);

m = mean(modeldata')';
% Subtract the mean from each image [Centering the data]
d=modeldata-repmat(m,1,tc);


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
% And also we can use the eigen vectors that the corresponding eigen values is greater than zero(Threshold) and this method will decrease the
% computation time and complixity

vec=eigvector(:,1:count1);

% Compute the feature matrix (the space that will use it to project the testing image on it)
train = double(traindata);

train = train - m;

train = vec'*train;
% If you have test data do the following
% this test data is close to the first class
test = double(testdata);
%Subtract the mean from the test data
test=test-m;
%Project the testing data on the space of the training data
test=vec'*test;

%Obtain euclidean distance between both training and testing
distance=pdist2(test',train','Euclidean');
zeroMatrix = zeros(5);
oneMatrix = ones(5);
labels = [];
labelCount = 1;
D = [];
for i=1:15 
    D = [];
    for j=1:15
        if(labelCount==j)
            D = [D,zeroMatrix];
        else
            D = [D,oneMatrix];
        end
    end
    labels = vertcat(labels,D);
    labelCount = labelCount + 1;
end

%Utilize ezroc function to evaluate performance
ezroc3(distance,labels,2,'',1);
