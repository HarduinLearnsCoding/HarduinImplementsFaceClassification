%% Main Script
% Evaluating the errors of different classifier based on the given Face
% recognition tasks


%% Data Processing

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

z=input("Enter the classification task (1 : Person or 2 : Face Type) \n");

switch z
    case 2
        NumofClasses=3;
    
%Classification Tasks : Person from image  and Neutral vs Expression
%Creating labels 
%Labels should increase for each row for first classification
%Labels should be 1,2,3,1,2,3 for second classification type
%% Struct manipulation

facetemp=face;

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


%% Training-Testing partition (Operating only on FACES rn)

x=randi([100 120]);  %Random Training set size 
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

[mean,covar]=standardestimators(trainingsorted,x,z);

[l,m]=size(covar(1).ClassCov);

[p,n]=size(covar);

%% Playing with covariances

for i=1:n
    lambda=0.92*ones(1,l);
    covar(i).ClassCov=covar(i).ClassCov + diag(lambda);
%     disp(det(covar(i).ClassCov));
end

%SINGULAR COVARIANCE


[m,~]=size(mean);
pseudoinv=[];

for i=1:m
    pseudoinv=[pseudoinv struct('Label', covar(i).Label, 'Data', pinv(covar(i).ClassCov))];
end


% pseudoinv=pinv(covar(3).ClassCov);
% disp(det(pseudoinv));

% disp([size(trainingsorted),size(testingsorted),x,Numtest]);

% disp(mean);
% disp(size(mean));
%% Bayes Classical


[classifiedBayes,values,error]=Bayes(testing,mean,covar,Numtest);

%ERROR BAYES FINAL

bayestesting=sprintf('Percentage of error for Bayes Classifier is %f',error);
disp(bayestesting);
%% KNN Classical


k=5;
Numtrain=200-Numtest;
[classifiedKNN,testingarray,trainingarray,distancetruncated,indextruncated,labels,actuallabels,errorknn]=KNN(testingsorted,trainingsorted,k,Numtest*3,600);

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



%% MDA(me) time

for n = 1:1:200
   facesMDAmean(n*3-2,1) = struct('Label', 1, 'Data', facesfinale(n).Neutral);
   facesMDAmean(n*3-1,1) = struct('Label', 2, 'Data', facesfinale(n).Expressive);
   facesMDAmean(n*3,1)   = struct('Label', 3, 'Data', facesfinale(n).Illumination);
end

tablenewnew = struct2table(facesMDAmean); 
sortedtableMDA = sortrows(tablenewnew, 'Label'); 
facesMDAmeansorted = table2struct(sortedtableMDA); 

[meanwhole,covarwhole]=standardestimators(facesMDAmeansorted,200,z);
[facesMDA,datavisualise,mean0,scatterbetween,scatterwithin,prior,eigenvalues,eigenvectors,eigenvaluesdiag,eigenvectorstr]=MDAsolver(facetemp,meanwhole,covarwhole,NumofClasses);
facesfinaleMDA=[];

for n=1:200
    facesfinaleMDA= cat(2,facesfinaleMDA,struct('Problem1Label',n,'Neutral',facesMDA(3*n-2,:),'Expressive',facesMDA(3*n-1,:),'Illumination',facesMDA(3*n,:)));
end

xMDA=x;  %Random Training set size 
NumtestMDA=200-xMDA;       %Random Testing set size

% training=vertcat(facesfinale.Neutral,facesfinale.Expressive,facesfinale.Illumination);
% The above statement can be used as well to create a column vector of all
% the feature vectors. The loop is resource consuming hence the following
% option has been chosen

% disp(x);

%1 represents Neutral, 2 Expressive and 3 Illuminated

for n = 1:1:xMDA
   trainingMDA(n*2-1,1) = struct('Label', 1, 'Data', (facesfinaleMDA(n).Neutral).');
   trainingMDA(n*2,1) = struct('Label', 2, 'Data', (facesfinaleMDA(n).Expressive).');
%    trainingMDA(n*3,1)   = struct('Label', 3, 'Data', (facesfinaleMDA(n).Illumination).');
end

n=1;
while n+xMDA<=200
   testingMDA(n*2-1,1) = struct('Label', 1, 'Data', (facesfinaleMDA(n+xMDA).Neutral).');
   testingMDA(n*2,1) = struct('Label', 2, 'Data', (facesfinaleMDA(n+xMDA).Expressive).');
%    testingMDA(n*3,1)   = struct('Label', 3, 'Data', (facesfinaleMDA(n+xMDA).Illumination).');
   n=n+1;
end

tablenew = struct2table(trainingMDA); 
sortedtable = sortrows(tablenew, 'Label'); 
trainingsortedMDA = table2struct(sortedtable); 

tablenewtest = struct2table(testingMDA); 
sortedtabletest = sortrows(tablenewtest, 'Label'); 
testingsortedMDA = table2struct(sortedtabletest); 

[meanMDA,covarMDA]=standardestimators(trainingsortedMDA,xMDA,z);

[l,~]=size(covarMDA(1).ClassCov);

[p,n]=size(covarMDA);

for i=1:n
    lambdaMDA=0.5*ones(1,l);
    covarMDA(i).ClassCov=covarMDA(i).ClassCov + diag(lambdaMDA);
%     disp(det(covar(i).ClassCov));
end

%SINGULAR COVARIANCE

[m,~]=size(meanMDA);
pseudoinvMDA=[];


for i=1:m
    pseudoinvMDA=[pseudoinvMDA struct('Label', covarMDA(i).Label, 'Data', pinv(covarMDA(i).ClassCov))];
end

%% Bayes MDA

% pseudoinv=pinv(covar(3).ClassCov);
% disp(det(pseudoinv));

% disp([size(trainingsorted),size(testingsorted),x,Numtest]);

% disp(mean);
% disp(size(mean));
% 
[classifiedBayesMDA,valuesMDA,errorMDA]=Bayes(testingMDA,meanMDA,covarMDA,NumtestMDA);

%ERROR BAYES FINAL

bayestestingMDA=sprintf('Percentage of error for Bayes Classifier with MDA is %f',errorMDA);
disp(bayestestingMDA);

%% KNN MDA

kMDA=5;
NumtrainMDA=200-NumtestMDA;
[classifiedKNNMDA,testingarrayMDA,trainingarrayMDA,distancetruncatedMDA,indextruncatedMDA,labelsMDA,actuallabelsMDA,errorknnMDA]=KNN(testingsortedMDA,trainingsortedMDA,1,NumtestMDA*2,400);
knntestingMDA=sprintf('Percentage of error for KNN Classifier with MDA is %f',errorknnMDA);
disp(knntestingMDA);

%% SVM trying

%Step 1 Make Kernels Done
%Step 2 Massage Data Done
%Step 3 G matrix creation

facesSVM=facetemp;
sigma=0.3;
facesfinaleSVM=[];
dimnface=504;

for n=1:200
    facesfinaleSVM= cat(2,facesfinaleSVM,struct('Problem1Label',n,'Neutral',reshape(face(:,:,3*n-2), [dimnface,1]),'Expressive',reshape(face(:,:,3*n-1), [dimnface,1]),'Illumination',reshape(face(:,:,3*n), [dimnface,1])));
end

xSVM=x;  %Random Training set size 
NumtestSVM=200-xSVM;       %Random Testing set size

%1 represents Neutral, 2 Expressive 

for n = 1:xSVM
   trainingSVM(n*2-1,1) = struct('Label', -1, 'Data', (facesfinaleMDA(n).Neutral).');
   trainingSVM(n*2,1) = struct('Label', 1, 'Data', (facesfinaleMDA(n).Expressive).');
end

n=1;
while n+xSVM<=200
   testingSVM(n*2-1,1) = struct('Label', -1, 'Data', (facesfinaleMDA(n+xSVM).Neutral).');
   testingSVM(n*2,1) = struct('Label', 1, 'Data', (facesfinaleMDA(n+xSVM).Expressive).');
   n=n+1;
end

tablenew = struct2table(trainingSVM); 
sortedtable = sortrows(tablenew, 'Label'); 
trainingsortedSVM = table2struct(sortedtable); 

tablenewtest = struct2table(testingSVM); 
sortedtabletest = sortrows(tablenewtest, 'Label'); 
testingsortedSVM = table2struct(sortedtabletest); 

[coltrain,~]=size(trainingSVM);
matrixG=Gmatrix(trainingSVM,sigma);
f=ones(coltrain,1);

for i=1:coltrain
    yvector(i,1)=trainingSVM(i).Label;
end

Aeq=yvector.';
beq=0;
lb=zeros(1,coltrain);
A=[];
b=[];

alpha = quadprog(matrixG,f,A,b,Aeq,beq,lb);

[coltest,~]=size(testingSVM);
PredictedLabelSum=zeros(coltest,1);

for i=1:coltest
    for j=1:coltrain
        PredictedLabelSum(i,1)=PredictedLabelSum(i,1)+alpha(j,1)*trainingSVM(j).Label*rbfkernel(trainingSVM(j).Data,testingSVM(i).Data,sigma);
    end
    PredictedLabelSum(i,1)=sign(PredictedLabelSum(i,1));
end

errorSVM=0;
errornew=0;
for i=1:coltest
    if PredictedLabelSum(i,1)~=testingSVM(i).Label
        errornew=errornew+1;
    end
end

errorSVM=(errornew/coltest)*100;
disp(errorSVM);

%%  PCA trying HUGE Error

PCAdim=10;

[facesPCA,covarPCAfn,eigenvaluesPCA,eigenvectorsPCA,eigenvaluesdiagPCA,eigenvectorstrPCA]=PCAsolver(facetemp,PCAdim);

facesfinalePCA=[];

for n=1:200
    facesfinalePCA= cat(2,facesfinalePCA,struct('Problem1Label',n,'Neutral',facesPCA(3*n-2,:),'Expressive',facesPCA(3*n-1,:),'Illumination',facesPCA(3*n,:)));
end

xPCA=x;  %Random Training set size 
NumtestPCA=200-xPCA;       %Random Testing set size

% training=vertcat(facesfinale.Neutral,facesfinale.Expressive,facesfinale.Illumination);
% The above statement can be used as well to create a column vector of all
% the feature vectors. The loop is resource consuming hence the following
% option has been chosen

% disp(x);

%1 represents Neutral, 2 Expressive and 3 Illuminated

for n = 1:1:xPCA
   trainingPCA(n*3-2,1) = struct('Label', 1, 'Data', (facesfinalePCA(n).Neutral).');
   trainingPCA(n*3-1,1) = struct('Label', 2, 'Data', (facesfinalePCA(n).Expressive).');
   trainingPCA(n*3,1)   = struct('Label', 3, 'Data', (facesfinalePCA(n).Illumination).');
end

n=1;
while n+xPCA<=200
   testingPCA(n*3-2,1) = struct('Label', 1, 'Data', (facesfinalePCA(n+xPCA).Neutral).');
   testingPCA(n*3-1,1) = struct('Label', 2, 'Data', (facesfinalePCA(n+xPCA).Expressive).');
   testingPCA(n*3,1)   = struct('Label', 3, 'Data', (facesfinalePCA(n+xPCA).Illumination).');
   n=n+1;
end

tablenew = struct2table(trainingPCA); 
sortedtable = sortrows(tablenew, 'Label'); 
trainingsortedPCA = table2struct(sortedtable); 

tablenewtest = struct2table(testingPCA); 
sortedtabletest = sortrows(tablenewtest, 'Label'); 
testingsortedPCA = table2struct(sortedtabletest); 

[meanPCA,covarPCA]=standardestimators(trainingsortedPCA,xPCA,z);

[l,~]=size(covarPCA(1).ClassCov);

[p,n]=size(covarPCA);

for i=1:n
    lambdaPCA=0.5*ones(1,l);
    covarPCA(i).ClassCov=covarPCA(i).ClassCov + diag(lambdaPCA);
%     disp(det(covar(i).ClassCov));
end

%SINGULAR COVARIANCE

[m,~]=size(meanPCA);
pseudoinvPCA=[];


for i=1:m
    pseudoinvPCA=[pseudoinvPCA struct('Label', covarMDA(i).Label, 'Data', pinv(covarPCA(i).ClassCov))];
end

%% Bayes PCA

[classifiedBayesPCA,valuesPCA,errorPCA]=Bayes(testingPCA,meanPCA,covarPCA,NumtestPCA);

%ERROR BAYES FINAL

bayestestingPCA=sprintf('Percentage of error for Bayes Classifier with PCA is %f',errorPCA);
disp(bayestestingPCA);

%% KNN PCA

kPCA=5;
NumtrainPCA=200-NumtestPCA;
[classifiedKNNPCA,testingarrayPCA,trainingarrayPCA,distancetruncatedPCA,indextruncatedPCA,labelsPCA,actuallabelsPCA,errorknnPCA]=KNN(testingsortedPCA,trainingsortedPCA,kPCA,NumtestPCA*3,600);
knntestingPCA=sprintf('Percentage of error for KNN Classifier with PCA is %f',errorknnPCA);
disp(knntestingPCA);

%% Problem 1 Trying

% [l,~]=size(trainingsorted(1).Data);
% [covarfixed]=regularise(covar,l);
% for i=1:3
%     disp(det(covarfixed(i).ClassCov));
% end
    case 1
        NumofClasses=200;

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
number=200;

facetemp=face;

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


for n = 1:1:number
   trainingP1(n*2-1,1) = struct('Label', n, 'Data', facesfinale(n).Neutral);
   trainingP1(n*2,1) = struct('Label', n, 'Data', facesfinale(n).Expressive);
end


for i=1:1:number
    testingP1(i,1)=struct('Label',i,'Data',facesfinale(i).Illumination);
end

[meanP1,covarP1]=standardestimators(trainingP1,number*2,1);

[l,m]=size(covarP1(1).ClassCov);

[p,n]=size(covarP1);

for i=1:n
    lambda=0.5*ones(1,l);
    covarP1(i).ClassCov=covarP1(i).ClassCov + diag(lambda);
%     disp(det(covar(i).ClassCov));
end

%SINGULAR COVARIANCE


[m,~]=size(meanP1);
pseudoinv=[];

Numtest=number;
%% Bayes P1

[classifiedBayesP1,valuesP1,errorP1]=Bayes(testingP1,meanP1,covarP1,Numtest);

%ERROR BAYES FINAL

bayestestingP1=sprintf('Percentage of error for Bayes Classifier is %f',errorP1);
disp(bayestestingP1);
%% KNN P1

k=5;
[classifiedKNN,testingarray,trainingarray,distancetruncated,indextruncated,labels,actuallabels,errorknn]=KNN(testingP1,trainingP1,k,Numtest);
knntesting=sprintf('Percentage of error for KNN Classifier is %f',errorknn);
disp(knntesting);

end


