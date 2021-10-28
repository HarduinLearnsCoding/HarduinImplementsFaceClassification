%% BASIC DATA PROCESSING

clc;
clear all;
close all;
load('./Datasets/data.mat');
load('./Datasets/illumination.mat');
load('./Datasets/pose.mat');
dimnface=24*21;
faces=zeros(dimnface,3,200);
facesfinale=[];
y=1:1:200;
NumofClasses=0;

z=input("Enter the classification task (1 : Face type or 2 : Person) \n");

switch z
    case 1
        NumofClasses=3;
    case 2
        NumofClasses=200;
end


%Classification Tasks : Person from image  and Neutral vs Expression
%Creating labels 
%Labels should increase for each row for first classification
%Labels should be 1,2,3,1,2,3 for second classification type
%% MDA Testing

facetemp=face;
% [facesMDA,datavisualise,mean0,scatterbetween,scatterwithin,prior,eigenvalues,eigenvectors,eigenvaluesdiag,eigenvectorstr]=MDAsolver(facetemp,mean,covar,NumofClasses);
% % disp(size(face));


for n=1:200
    facesfinale= cat(2,facesfinale,struct('Problem1Label',n,'Neutral',reshape(face(:,:,3*n-2), [dimnface,1]),'Expressive',reshape(face(:,:,3*n-1), [dimnface,1]),'Illumination',reshape(face(:,:,3*n), [dimnface,1])));
end

%disp(size(faces(:,1,:))); 504 1 200
% 
% disp(size(faces(:,1,:)))
% disp(size(faces(:,2,:)))
% disp(size(faces(:,3,:)))

dimnpose=48*40;
% size(pose)
posesfinale=[];

for n=1:68
        posesfinale= cat(1,posesfinale,struct('Image1',reshape(pose(:,:,1,n),[dimnpose,1]),'Image2',reshape(pose(:,:,2,n),[dimnpose,1]),'Image3',reshape(pose(:,:,3,n),[dimnpose,1]),'Image4',reshape(pose(:,:,4,n),[dimnpose,1]),'Image5',reshape(pose(:,:,5,n),[dimnpose,1]),'Image6',reshape(pose(:,:,6,n),[dimnpose,1]),'Image7',reshape(pose(:,:,7,n),[dimnpose,1]),'Image8',reshape(pose(:,:,8,n),[dimnpose,1]),'Image9',reshape(pose(:,:,9,n),[dimnpose,1]),'Image10',reshape(pose(:,:,10,n),[dimnpose,1]),'Image11',reshape(pose(:,:,11,n),[dimnpose,1]),'Image12',reshape(pose(:,:,12,n),[dimnpose,1]),'Image13',reshape(pose(:,:,13,n),[dimnpose,1])));
end

dimnillum=dimnpose;
illumsfinale=[];

for n=1:68
        illumsfinale= cat(1,illumsfinale,struct('Image1',reshape(reshape(illum(:,1,n), 48, 40),[dimnpose,1]),'Image2',reshape(reshape(illum(:,2,n), 48, 40),[dimnpose,1]),'Image3',reshape(reshape(illum(:,3,n), 48, 40),[dimnpose,1]),'Image4',reshape(reshape(illum(:,4,n), 48, 40),[dimnpose,1]),'Image5',reshape(reshape(illum(:,5,n), 48, 40),[dimnpose,1]),'Image6',reshape(reshape(illum(:,6,n), 48, 40),[dimnpose,1]),'Image7',reshape(reshape(illum(:,7,n), 48, 40),[dimnpose,1]),'Image8',reshape(reshape(illum(:,8,n), 48, 40),[dimnpose,1]),'Image9',reshape(reshape(illum(:,9,n), 48, 40),[dimnpose,1]),'Image10',reshape(reshape(illum(:,10,n), 48, 40),[dimnpose,1]),'Image11',reshape(reshape(illum(:,11,n), 48, 40),[dimnpose,1]),'Image12',reshape(reshape(illum(:,12,n), 48, 40),[dimnpose,1]),'Image13',reshape(reshape(illum(:,13,n), 48, 40),[dimnpose,1]),'Image14',reshape(reshape(illum(:,14,n), 48, 40),[dimnpose,1]),'Image15',reshape(reshape(illum(:,15,n), 48, 40),[dimnpose,1]),'Image16',reshape(reshape(illum(:,16,n), 48, 40),[dimnpose,1]),'Image17',reshape(reshape(illum(:,17,n), 48, 40),[dimnpose,1]),'Image18',reshape(reshape(illum(:,18,n), 48, 40),[dimnpose,1]),'Image19',reshape(reshape(illum(:,19,n), 48, 40),[dimnpose,1]),'Image20',reshape(reshape(illum(:,20,n), 48, 40),[dimnpose,1]),'Image21',reshape(reshape(illum(:,21,n), 48, 40),[dimnpose,1])));
end


%% BREAK INTO TRAINING AND TESTING (Operating only on FACES rn)

x=randi([160 165]);  %Random Training set size 
Numtest=200-x;       %Random Testing set size

% training=vertcat(facesfinale.Neutral,facesfinale.Expressive,facesfinale.Illumination);
% The above statement can be used as well to create a column vector of all
% the feature vectors. The loop is resource consuming hence the following
% option has been chosen

% disp(x);

%1 represents Neutral, 2 Expressive and 3 Illuminated

for n = 1:1:x
   training(n*3-2,1) = struct('Label', 1, 'Data', facesfinale(n).Neutral);
   training(n*3-1,1) = struct('Label', 2, 'Data', facesfinale(n).Expressive);
   training(n*3,1)   = struct('Label', 3, 'Data', facesfinale(n).Illumination);
end

n=1;
while n+x<=200
   testing(n*3-2,1) = struct('Label', 1, 'Data', facesfinale(n+x).Neutral);
   testing(n*3-1,1) = struct('Label', 2, 'Data', facesfinale(n+x).Expressive);
   testing(n*3,1)   = struct('Label', 3, 'Data', facesfinale(n+x).Illumination);
   n=n+1;
end

% for n = 1:1:Numtest
%    testing(n*3-2,1) = struct('Label', 1, 'Data', facesfinale(n+x).Neutral);
%    testing(n*3-1,1) = struct('Label', 2, 'Data', facesfinale(n+x).Expressive);
%    testing(n*3,1)   = struct('Label', 3, 'Data', facesfinale(n+x).Illumination);
% end

%Sorting the labels

tablenew = struct2table(training); 
sortedtable = sortrows(tablenew, 'Label'); 
trainingsorted = table2struct(sortedtable); 

tablenewtest = struct2table(testing); 
sortedtabletest = sortrows(tablenewtest, 'Label'); 
testingsorted = table2struct(sortedtabletest); 

[mean,covar]=standardestimators(trainingsorted,x,1);

[l,m]=size(covar(1).ClassCov);

[p,n]=size(covar);

for i=1:n
    lambda=0.5*ones(1,l);
    covar(i).ClassCov=covar(i).ClassCov + diag(lambda);
%     disp(det(covar(i).ClassCov));
end

%SINGULAR COVARIANCE

[m,~]=size(mean);
pseudoinv=[];

for i=1:m
    pseudoinv=[pseudoinv struct('Label', covar(i).Label, 'Data', pinv(covar(i).ClassCov))];
end

%% MDA AND PCA TIME

[facesMDA,datavisualise,mean0,scatterbetween,scatterwithin,prior,eigenvalues,eigenvectors,eigenvaluesdiag,eigenvectorstr]=MDAsolver(facetemp,mean,covar,NumofClasses);

%% Maybe useful deleted stuff

% pseudoinv=pinv(covar(3).ClassCov);
% disp(det(pseudoinv));

% disp([size(trainingsorted),size(testingsorted),x,Numtest]);

% disp(mean);
% disp(size(mean));
% 
[classifiedBayes,values,error]=Bayes(testing,mean,covar,Numtest);

%ERROR BAYES FINAL

bayestesting=sprintf('Percentage of error for Bayes Classifier is %f',error);
disp(bayestesting);
%% KNN TRYING


k=5;
Numtrain=200-Numtest;
[classifiedKNN,testingarray,trainingarray,distancetruncated,indextruncated,labels,actuallabels,errorknn]=KNN(testingsorted,trainingsorted,k,Numtest);

% for i=1:Numtest*3
% 
%     testingarray(i,:)=testingsorted(i).Data.';
%     t_labels(i,:)=testingsorted(i).Label;
% 
% end
% 
% for j=1:Numtrain*3
%     
%     trainingarray(j,:)=trainingsorted(j).Data.';
%     labels(j,:)=trainingsorted(j).Label;
% 
% end
% 
% [~,~,accuracy]=KNN_(k,trainingarray,labels,testingarray,t_labels);
% disp(accuracy);
knntesting=sprintf('Percentage of error for KNN Classifier is %f',errorknn);
disp(knntesting);

% [l,~]=size(trainingsorted(1).Data);
% [covarfixed]=regularise(covar,l);
% for i=1:3
%     disp(det(covarfixed(i).ClassCov));
% end


% %% PROBLEM1 Labels?
% % 
% % x=randi([100 150]);  %Random Training set size 
% % Numtest=200-x;       %Random Testing set size
% 
% % training=vertcat(facesfinale.Neutral,facesfinale.Expressive,facesfinale.Illumination);
% % The above statement can be used as well to create a column vector of all
% % the feature vectors. The loop is resource consuming hence the following
% % option has been chosen
% 
% % disp(x);
% 
% number=200;
% 
% 
% for n = 1:1:number
%    trainingP1(n*2-1,1) = struct('Label', n, 'Data', facesfinale(n).Neutral);
%    trainingP1(n*2,1) = struct('Label', n, 'Data', facesfinale(n).Expressive);
% end
% 
% 
% for i=1:1:number
%     testingP1(i,1)=struct('Label',n,'Data',facesfinale(n).Illumination);
% end

