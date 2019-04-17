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
traindata = [];
testdata = [];
for i = 1 : 40
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

[tr,tc] = size(traindata);
traindata = double(traindata);

m = mean(traindata')';
% Subtract the mean from each image [Centering the data]
d=traindata-repmat(m,1,tc);


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
x=vec'*d;

% If you have test data do the following
% this test data is close to the first class
t = double(testdata);
%Subtract the mean from the test data
t=t-m;
%Project the testing data on the space of the training data
t=vec'*t;

%Obtain euclidean distance between both training and testing
distance=pdist2(t',x','Euclidean');
zeroMatrix = zeros(5);
oneMatrix = ones(5);
labels = [];
labelCount = 1;
for i=1:40 
    D = [];
    for j=1:40
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