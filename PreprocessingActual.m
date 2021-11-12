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
   training(n*2-1,1) = struct('Label', 1, 'Data', facesfinale(n).Neutral);
   training(n*2,1) = struct('Label', 2, 'Data', facesfinale(n).Expressive);
end

n=1;
while n+x<=200
   testing(n*2-1,1) = struct('Label', 1, 'Data', facesfinale(n+x).Neutral);
   testing(n*2,1) = struct('Label', 2, 'Data', facesfinale(n+x).Expressive);
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


[classifiedBayes,values,error]=Bayes(testingsorted,mean,covar,Numtest*2);
bayestesting=sprintf('\nPercentage of error for Bayes Classifier is %f',error);
disp(bayestesting);

%% KNN Classical


k=5;
Numtrain=200-Numtest;
[classifiedKNN,testingarray,trainingarray,distancetruncated,indextruncated,labels,actuallabels,errorknn]=KNN(testingsorted,trainingsorted,k,Numtest*2,400);
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
[facesMDA,datavisualise,mean0,scatterbetween,scatterwithin,prior,eigenvalues,eigenvectors,eigenvaluesdiag,eigenvectorstr]=MDAsolver(facetemp,meanwhole,covarwhole,NumofClasses,NumofClasses);
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

% 
[classifiedBayesMDA,valuesMDA,errorMDA]=Bayes(testingMDA,meanMDA,covarMDA,NumtestMDA*2);

%ERROR BAYES FINAL

bayestestingMDA=sprintf('Percentage of error for Bayes Classifier with MDA is %f',errorMDA);
disp(bayestestingMDA);

%% KNN MDA

kMDA=5;
NumtrainMDA=200-NumtestMDA;
[classifiedKNNMDA,testingarrayMDA,trainingarrayMDA,distancetruncatedMDA,indextruncatedMDA,labelsMDA,actuallabelsMDA,errorknnMDA]=KNN(testingsortedMDA,trainingsortedMDA,1,NumtestMDA*2,400);
knntestingMDA=sprintf('Percentage of error for KNN Classifier with MDA is %f',errorknnMDA);
disp(knntestingMDA);


%% SVM Done

%Step 1 Make Kernels Done
%Step 2 Massage Data Done
%Step 3 G matrix creation Done

kerneltype=input(' \nEnter the kernel type (1: Linear Kernel 2: RBF Kernel 3: Polynomial Kernel) \n');

facesSVM=facetemp;
sigma=0.6;
facesfinaleSVM=[];
dimnface=504;
r=20;

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
matrixG=Gmatrix(trainingSVM,sigma,kerneltype,r);
f=ones(coltrain,1);

for i=1:coltrain
    yvector(i,1)=trainingSVM(i).Label;
end

Aeq=yvector.';
beq=0;
lb=zeros(1,coltrain);
A=[];
b=[];
x0=[];
ub=[];

options = optimset('Display', 'off');
alpha = quadprog(matrixG,f,A,b,Aeq,beq,lb,ub,x0,options);

[coltest,~]=size(testingSVM);
PredictedLabelSum=zeros(coltest,1);

switch kerneltype
    case 2
    for i=1:coltest
        for j=1:coltrain
            PredictedLabelSum(i,1)=PredictedLabelSum(i,1)+alpha(j,1)*trainingSVM(j).Label*rbfkernel(trainingSVM(j).Data,testingSVM(i).Data,sigma);
        end
        PredictedLabelSum(i,1)=sign(PredictedLabelSum(i,1));
    end
    case 1
        for i=1:coltest
            for j=1:coltrain
                PredictedLabelSum(i,1)=PredictedLabelSum(i,1)+alpha(j,1)*trainingSVM(j).Label*linearkernel(trainingSVM(j).Data,testingSVM(i).Data);
            end
        PredictedLabelSum(i,1)=sign(PredictedLabelSum(i,1));
    end
    case 3
        for i=1:coltest
            for j=1:coltrain
                PredictedLabelSum(i,1)=PredictedLabelSum(i,1)+alpha(j,1)*trainingSVM(j).Label*polynomialkernel(trainingSVM(j).Data,testingSVM(i).Data,r);
            end
        PredictedLabelSum(i,1)=sign(PredictedLabelSum(i,1));
    end
end


errorSVM=0;
errornew=0;
for i=1:coltest
    if PredictedLabelSum(i,1)~=testingSVM(i).Label
        errornew=errornew+1;
    end
end

errorSVM=(errornew/coltest)*100;
errorSVM=sprintf('\nPercentage of error for Kernel SVM Classifier with MDA (without any cross validation) is %f',errorSVM);
disp(errorSVM);

%% K fold CV (Time consuming) Commented out?

% k=3;
% 
% %For high k the program runs for a long time.
% 
% %For the polynomial kernel, the program runs for a longer time than the RBF
% 
% kerneltype=input(' \nEnter the kernel type (2: RBF Kernel 3: Polynomial Kernel) \n');
% c = cvpartition(200,'KFold',k);
% trainingdataKfold=[];
% testingdataKfold=[];
% errorSVM=0;
% count=0;
% PredictedLabelSum=[];
% setiter=[0.1 0.7 1 3 15 50];
% iterationvar=exp(0.05):exp(1):exp(3.05);
% parameterval=0;
% avgerror=ones(5,1);
% 
% countvar=1;
% 
% switch kerneltype
%     
%     case 2
%         
% for variable=[0.05 1.05 3.05 20]
%     errorSVM=0;
%     count=0;
%     errorSVMiter=0;
%     
% for i=1:k
%     idxTrain = c.training(k) ;
%     idxTest = c.test(k);
%     
%     for j=1:200
%         if idxTrain(j)==1
%             trainingdataKfold=[trainingdataKfold struct('Neutral',facesfinaleMDA(j).Neutral,'Expressive',facesfinaleMDA(j).Expressive)];
%         end
%         if idxTest(j)==1
%             testingdataKfold=[testingdataKfold struct('Neutral',facesfinaleMDA(j).Neutral,'Expressive',facesfinaleMDA(j).Expressive)];
%         end
%     end
%     
%     [kfoldrow,kfoldcol]=size(trainingdataKfold);
%     
%     for n = 1:kfoldcol
%         trainingSVMkfold(n*2-1,1) = struct('Label', -1, 'Data', (trainingdataKfold(n).Neutral).');
%         trainingSVMkfold(n*2,1) = struct('Label', 1, 'Data', (trainingdataKfold(n).Expressive).');
%     end
%     
%     [kfoldrowtest,kfoldcoltest]=size(testingdataKfold);
%     
%     for l = 1:kfoldcoltest
%         testingSVMkfold(l*2-1,1) = struct('Label', -1, 'Data', (testingdataKfold(l).Neutral).');
%         testingSVMkfold(l*2,1) = struct('Label', 1, 'Data', (testingdataKfold(l).Expressive).');
%     end
%     
%     tablenew = struct2table(trainingSVMkfold); 
%     sortedtable = sortrows(tablenew, 'Label'); 
%     trainingsortedSVM = table2struct(sortedtable); 
% 
%     tablenewtest = struct2table(testingSVMkfold); 
%     sortedtabletest = sortrows(tablenewtest, 'Label'); 
%     testingsortedSVM = table2struct(sortedtabletest); 
% 
% [coltrain,~]=size(trainingSVMkfold);
% 
% [coltest,~]=size(testingSVMkfold);
% 
% 
% [PredictedLabelSum,errorSVM]=KfoldKernelSVMsolver(trainingSVMkfold,testingSVMkfold,kerneltype,variable,variable);
% 
% errorSVMiter=errorSVMiter+errorSVM;
%     
% end
% 
% countvar=countvar+1;
% 
% avgerror(countvar)=errorSVMiter/k;
%    
% disp('Please Wait');
% end
% [~,idxparam]=min(avgerror(2:5));
% allparam=[0.05 1.05 3.05 20];
% parameterval=allparam(idxparam);
% 
% optimalparam=sprintf('\nOptimal Sigma from initialised set is %f',parameterval);
% disp(optimalparam);
% 
%     case 3
%         
% for variable=[1.05 20]
%     errorSVM=0;
%     count=0;
%     errorSVMiter=0;
%     
% for i=1:k
%     idxTrain = c.training(k) ;
%     idxTest = c.test(k);
%     
%     for j=1:200
%         if idxTrain(j)==1
%             trainingdataKfold=[trainingdataKfold struct('Neutral',facesfinaleMDA(j).Neutral,'Expressive',facesfinaleMDA(j).Expressive)];
%         end
%         if idxTest(j)==1
%             testingdataKfold=[testingdataKfold struct('Neutral',facesfinaleMDA(j).Neutral,'Expressive',facesfinaleMDA(j).Expressive)];
%         end
%     end
%     
%     [kfoldrow,kfoldcol]=size(trainingdataKfold);
%     
%     for n = 1:kfoldcol
%         trainingSVMkfold(n*2-1,1) = struct('Label', -1, 'Data', (trainingdataKfold(n).Neutral).');
%         trainingSVMkfold(n*2,1) = struct('Label', 1, 'Data', (trainingdataKfold(n).Expressive).');
%     end
%     
%     [kfoldrowtest,kfoldcoltest]=size(testingdataKfold);
%     
%     for l = 1:kfoldcoltest
%         testingSVMkfold(l*2-1,1) = struct('Label', -1, 'Data', (testingdataKfold(l).Neutral).');
%         testingSVMkfold(l*2,1) = struct('Label', 1, 'Data', (testingdataKfold(l).Expressive).');
%     end
%     
%     tablenew = struct2table(trainingSVMkfold); 
%     sortedtable = sortrows(tablenew, 'Label'); 
%     trainingsortedSVM = table2struct(sortedtable); 
% 
%     tablenewtest = struct2table(testingSVMkfold); 
%     sortedtabletest = sortrows(tablenewtest, 'Label'); 
%     testingsortedSVM = table2struct(sortedtabletest); 
% 
% [coltrain,~]=size(trainingSVMkfold);
% 
% [coltest,~]=size(testingSVMkfold);
% 
% 
% [PredictedLabelSum,errorSVM]=KfoldKernelSVMsolver(trainingSVMkfold,testingSVMkfold,kerneltype,variable,variable);
% 
% errorSVMiter=errorSVMiter+errorSVM;
%     
% end
% 
% countvar=countvar+1;
% 
% avgerror(countvar)=errorSVMiter/k;
% 
% disp('Please wait');
% end
% [~,idxparam]=min(avgerror(2:3));
% allparam=[1.05 20];
% parameterval=allparam(idxparam);
% 
% optimalparam=sprintf('\nOptimal r from initialised set is %f',parameterval);
% disp(optimalparam);
% end

%% Adaboost with Linear SVM (Takes time)

filler=10;
NumClassifiers=100;
iterations=NumClassifiers;
[datasetSVMsize,~]=size(trainingSVM);
coefficient=zeros(iterations,1);
[trainingSVMsamplesize,~]=size(trainingSVM(1).Data);
saveepsilon=zeros(iterations,1);
weightwithiteration=ones(datasetSVMsize,1);
weightwithiteration=weightwithiteration/datasetSVMsize;
savingprivateryan=[];

for iter=1:iterations
 
        probabilitymatrix=weightwithiteration/sum(weightwithiteration.');
        
        tempcount=1:datasetSVMsize;
        
        selectedsubset=randsample(tempcount,ceil(2/3*datasetSVMsize),true,probabilitymatrix);

        trainingSVMboost = trainingSVM(selectedsubset);
        
        savingprivateryan=[savingprivateryan struct('Data',trainingSVMboost)];
            
        [coltrain,~]=size(trainingSVMboost);
            
        matrixG=Gmatrix(trainingSVMboost,filler,1,filler);
            
        f=ones(coltrain,1);

        for i=1:coltrain
                
              yvectorboost(i,1)=trainingSVMboost(i).Label;
                
        end

            Aeq=yvectorboost.';
            beq=0;
            lb=zeros(1,coltrain);
            A=[];
            b=[];
            x0=[];
            ub=[];

            options = optimset('Display', 'off');
            alphaboost = quadprog(matrixG,f,A,b,Aeq,beq,lb,ub,x0,options);

            [coltest,~]=size(trainingSVM);
            PredictedLabelSumAdaboost=zeros(coltest,1);
    
            for i=1:coltest
                
                for j=1:coltrain
                    
                    PredictedLabelSumAdaboost(i,1)=PredictedLabelSumAdaboost(i,1)+alphaboost(j,1)*trainingSVMboost(j).Label*linearkernel(trainingSVMboost(j).Data,trainingSVM(i).Data);
                
                end
                
            PredictedLabelSumAdaboost(i,1)=sign(PredictedLabelSumAdaboost(i,1));
            
            end
            
            errorsummation=0;
            
            for i=1:coltest
                
                errornewboost=0;
                
                if PredictedLabelSumAdaboost(i,1)~=trainingSVM(i).Label
                    
                    errornewboost=errornewboost+1;
                    
                end
                
                errornewboost=probabilitymatrix(i,1)*errornewboost;
                
                errorsummation= errorsummation + errornewboost;
                
            end
            
            epsilon=0;
            
            for i=1:coltest
                
                if PredictedLabelSumAdaboost(i,1)~=trainingSVM(i).Label
            
                    epsilon=epsilon+probabilitymatrix(i);
                
                end
                
            
            end
            
            saveepsilon(iter)=epsilon;
            
            coefficient(iter)=0.5*log((1-epsilon)/epsilon);
            
            for i=1:coltest
            
                weightwithiteration(i,1)=weightwithiteration(i,1)*exp(-trainingSVM(i).Label*coefficient(iter)*PredictedLabelSumAdaboost(i,1));

            end   
                     
end

[SVMtestsize,~]=size(testingSVM);
Adaboosttestclassifier=zeros(SVMtestsize,1);
ClassifierX=[];
[PredictedLabelsSVMfn]=AdaSVMsolver(trainingSVM,testingSVM);


for iter=1:iterations
        ClassifierX=AdaSVMsolver(savingprivateryan(iter).Data,testingSVM);
        Adaboosttestclassifier=Adaboosttestclassifier+coefficient(iter)*ClassifierX;
end
Adaboosttestclassifier=sign(Adaboosttestclassifier);
    
[sizeclassifiedada,~]=size(Adaboosttestclassifier);

error=0;

for i=1:sizeclassifiedada
    
    if Adaboosttestclassifier(i,1)~=testingSVM(i).Label
        
        error=error+1;
    
    end
    
    
end

errorada=sprintf('Percentage of error for Linear SVM Classifier with Adaboost is %f',(error/sizeclassifiedada)*100);
disp(errorada);


%%  PCA reduction

PCAdim=300;

dimnface=504;

[lface,bface,Observationsize]=size(face);

PCAdatacenter=zeros(Observationsize,dimnface);

PCAdatacentered=zeros(Observationsize,dimnface);

meanPCAcenter=0;

for i=1:Observationsize
    PCAdatacenter(i,:)=reshape(face(:,:,i),[dimnface,1]);
end

% for i=1:dimnface
%     PCAdatacentered(:,i)=PCAdatacenter(:,i)-((sum(PCAdatacenter(:,i))/Observationsize)*ones(Observationsize,1));
% end

[facesPCA,covarPCAfn,eigenvaluesPCA,eigenvectorsPCA,eigenvaluesdiagPCA,eigenvectorstrPCA]=PCAsolver(PCAdatacenter,PCAdim);

facesfinalePCA=[];

for n=1:200
    facesfinalePCA= cat(2,facesfinalePCA,struct('Problem1Label',n,'Neutral',facesPCA(3*n-2,:),'Expressive',facesPCA(3*n-1,:),'Illumination',facesPCA(3*n,:)));
end

xPCA=x;  %Random Training set size 
NumtestPCA=200-xPCA;       %Random Testing set size

for n = 1:1:xPCA
   trainingPCA(n*2-1,1) = struct('Label', 1, 'Data', (facesfinalePCA(n).Neutral).');
   trainingPCA(n*2,1) = struct('Label', 2, 'Data', (facesfinalePCA(n).Expressive).');
%    trainingPCA(n*3,1)   = struct('Label', 3, 'Data', (facesfinalePCA(n).Illumination).');
end

n=1;
while n+xPCA<=200
   testingPCA(n*2-1,1) = struct('Label', 1, 'Data', (facesfinalePCA(n+xPCA).Neutral).');
   testingPCA(n*2,1) = struct('Label', 2, 'Data', (facesfinalePCA(n+xPCA).Expressive).');
%    testingPCA(n*3,1)   = struct('Label', 3, 'Data', (facesfinalePCA(n+xPCA).Illumination).');
   n=n+1;
end

tablenew = struct2table(trainingPCA); 
sortedtable = sortrows(tablenew, 'Label'); 
trainingsortedPCA = table2struct(sortedtable); 

tablenewtest = struct2table(testingPCA); 
sortedtabletest = sortrows(tablenewtest, 'Label'); 
testingsortedPCA = table2struct(sortedtabletest); 

[meanPCA,covarPCA]=standardestimators(trainingsortedPCA,xPCA,1);

[l,~]=size(covarPCA(1).ClassCov);

[p,n]=size(covarPCA);

for i=1:n
    lambdaPCA=0.5*ones(1,l);
    covarPCA(i).ClassCov=covarPCA(i).ClassCov + diag(lambdaPCA);
end


%% Bayes PCA

[classifiedBayesPCA,valuesPCA,errorPCA]=Bayes(testingsortedPCA,meanPCA,covarPCA,NumtestPCA*2);
bayestestingPCA=sprintf('Percentage of error for Bayes Classifier with PCA is %f',errorPCA);
disp(bayestestingPCA);

%% KNN PCA

kPCA=3;
[classifiedKNNPCA,testingarrayPCA,trainingarrayPCA,distancetruncatedPCA,indextruncatedPCA,labelsPCA,actuallabelsPCA,errorknnPCA]=KNN(testingsortedPCA,trainingsortedPCA,kPCA,NumtestPCA*2,400);
knntestingPCA=sprintf('Percentage of error for KNN Classifier with PCA is %f',errorknnPCA);
disp(knntestingPCA);

%% Classification task 1 Data Dataset
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

number=200;
facetemp=face;
dimensions=number;

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
% illumsfinale=[];
% 
% for n=1:68
%         illumsfinale= cat(1,illumsfinale,struct('Image1',reshape(reshape(illum(:,1,n), 48, 40),[dimnpose,1]),'Image2',reshape(reshape(illum(:,2,n), 48, 40),[dimnpose,1]),'Image3',reshape(reshape(illum(:,3,n), 48, 40),[dimnpose,1]),'Image4',reshape(reshape(illum(:,4,n), 48, 40),[dimnpose,1]),'Image5',reshape(reshape(illum(:,5,n), 48, 40),[dimnpose,1]),'Image6',reshape(reshape(illum(:,6,n), 48, 40),[dimnpose,1]),'Image7',reshape(reshape(illum(:,7,n), 48, 40),[dimnpose,1]),'Image8',reshape(reshape(illum(:,8,n), 48, 40),[dimnpose,1]),'Image9',reshape(reshape(illum(:,9,n), 48, 40),[dimnpose,1]),'Image10',reshape(reshape(illum(:,10,n), 48, 40),[dimnpose,1]),'Image11',reshape(reshape(illum(:,11,n), 48, 40),[dimnpose,1]),'Image12',reshape(reshape(illum(:,12,n), 48, 40),[dimnpose,1]),'Image13',reshape(reshape(illum(:,13,n), 48, 40),[dimnpose,1]),'Image14',reshape(reshape(illum(:,14,n), 48, 40),[dimnpose,1]),'Image15',reshape(reshape(illum(:,15,n), 48, 40),[dimnpose,1]),'Image16',reshape(reshape(illum(:,16,n), 48, 40),[dimnpose,1]),'Image17',reshape(reshape(illum(:,17,n), 48, 40),[dimnpose,1]),'Image18',reshape(reshape(illum(:,18,n), 48, 40),[dimnpose,1]),'Image19',reshape(reshape(illum(:,19,n), 48, 40),[dimnpose,1]),'Image20',reshape(reshape(illum(:,20,n), 48, 40),[dimnpose,1]),'Image21',reshape(reshape(illum(:,21,n), 48, 40),[dimnpose,1])));
% end



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

Numtest=number;

for n = 1:1:200
   facesMDAmean(n*3-2,1) = struct('Label', n, 'Data', facesfinale(n).Neutral);
   facesMDAmean(n*3-1,1) = struct('Label', n, 'Data', facesfinale(n).Expressive);
   facesMDAmean(n*3,1)   = struct('Label', n, 'Data', facesfinale(n).Illumination);
end

tablenewnew = struct2table(facesMDAmean); 
sortedtableMDA = sortrows(tablenewnew, 'Label'); 
facesMDAmeansorted = table2struct(sortedtableMDA); 

[meanwhole,covarwhole]=standardestimators(facesMDAmeansorted,200,z);
[facesMDA,datavisualise,mean0,scatterbetween,scatterwithin,prior,eigenvalues,eigenvectors,eigenvaluesdiag,eigenvectorstr]=MDAsolver(facetemp,meanwhole,covarwhole,NumofClasses,dimensions);
facesfinaleMDA=[];

for n=1:200
    facesfinaleMDA= cat(2,facesfinaleMDA,struct('Problem1Label',n,'Neutral',facesMDA(3*n-2,:),'Expressive',facesMDA(3*n-1,:),'Illumination',facesMDA(3*n,:)));
end

xMDA=200;  
NumtestMDA=200;       

for n = 1:1:xMDA
   trainingMDA(n*2-1,1) = struct('Label', n, 'Data', (facesfinaleMDA(n).Neutral).');
   trainingMDA(n*2,1) = struct('Label', n, 'Data', (facesfinaleMDA(n).Expressive).');
end

n=1;
while n<=200
   testingMDA(n,1) = struct('Label', n, 'Data', (facesfinaleMDA(n).Illumination).');
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

%% Bayes P1

[classifiedBayesP1,valuesP1,errorP1]=Bayes(testingMDA,meanMDA,covarMDA,NumtestMDA);

%ERROR BAYES FINAL

bayestestingP1=sprintf('Percentage of error for Bayes Classifier is %f',errorP1);
disp(bayestestingP1);
%% KNN P1

k=1;
[classifiedKNN,testingarray,trainingarray,distancetruncated,indextruncated,labels,actuallabels,errorknn]=KNN(testingMDA,trainingMDA,k,NumtestMDA,600);
knntesting=sprintf('Percentage of error for KNN Classifier is %f',errorknn);
disp(knntesting);

%% Classification Task 1 Pose Dataset

posessamples=68;
dimnpose=48*40;
% size(pose)
posesfinale=[];


for n=1:68
        posesfinale= cat(1,posesfinale,struct('P1Label',n,'Image1',reshape(pose(:,:,1,n),[dimnpose,1]),'Image2',reshape(pose(:,:,2,n),[dimnpose,1]),'Image3',reshape(pose(:,:,3,n),[dimnpose,1]),'Image4',reshape(pose(:,:,4,n),[dimnpose,1]),'Image5',reshape(pose(:,:,5,n),[dimnpose,1]),'Image6',reshape(pose(:,:,6,n),[dimnpose,1]),'Image7',reshape(pose(:,:,7,n),[dimnpose,1]),'Image8',reshape(pose(:,:,8,n),[dimnpose,1]),'Image9',reshape(pose(:,:,9,n),[dimnpose,1]),'Image10',reshape(pose(:,:,10,n),[dimnpose,1]),'Image11',reshape(pose(:,:,11,n),[dimnpose,1]),'Image12',reshape(pose(:,:,12,n),[dimnpose,1]),'Image13',reshape(pose(:,:,13,n),[dimnpose,1])));
end

for n = 1:1:posessamples
   trainingP1pose(n*8-7,1) = struct('Label', n, 'Data', posesfinale(n).Image1);
   trainingP1pose(n*8-6,1) = struct('Label', n, 'Data', posesfinale(n).Image2);
   trainingP1pose(n*8-5,1) = struct('Label', n, 'Data', posesfinale(n).Image3);
   trainingP1pose(n*8-4,1) = struct('Label', n, 'Data', posesfinale(n).Image4);
   trainingP1pose(n*8-3,1) = struct('Label', n, 'Data', posesfinale(n).Image5);
   trainingP1pose(n*8-2,1) = struct('Label', n, 'Data', posesfinale(n).Image6);
   trainingP1pose(n*8-1,1) = struct('Label', n, 'Data', posesfinale(n).Image7);
   trainingP1pose(n*8,1) = struct('Label', n, 'Data', posesfinale(n).Image8);
end


for i=1:1:posessamples
    testingP1pose(5*i-4,1)=struct('Label',i,'Data',posesfinale(i).Image9);
    testingP1pose(5*i-3,1)=struct('Label',i,'Data',posesfinale(i).Image10);
    testingP1pose(5*i-2,1)=struct('Label',i,'Data',posesfinale(i).Image11);
    testingP1pose(5*i-1,1)=struct('Label',i,'Data',posesfinale(i).Image12);
    testingP1pose(5*i,1)=struct('Label',i,'Data',posesfinale(i).Image13);
end

[meanP1pose,covarP1pose]=standardestimators(trainingP1pose,posessamples*8,1);

[l,m]=size(covarP1pose(1).ClassCov);

[p,n]=size(covarP1pose);

for i=1:n
    lambda=0.5*ones(1,l);
    covarP1pose(i).ClassCov=covarP1pose(i).ClassCov + diag(lambda);
%     disp(det(covar(i).ClassCov));
end

Numtest=posessamples;

for n = 1:1:68
   wholeP1poseMDA(n*13-12,1) = struct('Label', n, 'Data', posesfinale(n).Image1);
   wholeP1poseMDA(n*13-11,1) = struct('Label', n, 'Data', posesfinale(n).Image2);
   wholeP1poseMDA(n*13-10,1) = struct('Label', n, 'Data', posesfinale(n).Image3);
   wholeP1poseMDA(n*13-9,1) = struct('Label', n, 'Data', posesfinale(n).Image4);
   wholeP1poseMDA(n*13-8,1) = struct('Label', n, 'Data', posesfinale(n).Image5);
   wholeP1poseMDA(n*13-7,1) = struct('Label', n, 'Data', posesfinale(n).Image6);
   wholeP1poseMDA(n*13-6,1) = struct('Label', n, 'Data', posesfinale(n).Image7);
   wholeP1poseMDA(n*13-5,1) = struct('Label', n, 'Data',   posesfinale(n).Image8);
   wholeP1poseMDA(n*13-4,1) = struct('Label', n, 'Data', posesfinale(n).Image9);
   wholeP1poseMDA(n*13-3,1) = struct('Label', n, 'Data', posesfinale(n).Image10);
   wholeP1poseMDA(n*13-2,1) = struct('Label', n, 'Data', posesfinale(n).Image11);
   wholeP1poseMDA(n*13-1,1) = struct('Label', n, 'Data', posesfinale(n).Image12);
   wholeP1poseMDA(n*13,1) = struct('Label', n, 'Data', posesfinale(n).Image13);
end

posetemp=pose;
dimensionsposes=67;

tablenewnew = struct2table(wholeP1poseMDA); 
sortedtableMDA = sortrows(tablenewnew, 'Label'); 
posesMDAmeansorted = table2struct(sortedtableMDA); 

[meanwholeMDA,covarwholeMDA]=standardestimators(posesMDAmeansorted,68,z);
[posesMDA,datavisualise,mean0,scatterbetween,scatterwithin,prior,eigenvalues,eigenvectors,eigenvaluesdiag,eigenvectorstr]=poseMDAsolver(posetemp,meanwholeMDA,covarwholeMDA,68,dimensionsposes);
posesfinaleMDA=[];

for n=1:68
      posesfinaleMDA= cat(1,posesfinaleMDA,struct('Image1',posesMDA(:,:,1,n),'Image2',posesMDA(:,:,2,n),'Image3',posesMDA(:,:,3,n),'Image4',posesMDA(:,:,4,n),'Image5',posesMDA(:,:,5,n),'Image6',posesMDA(:,:,6,n),'Image7',posesMDA(:,:,7,n),'Image8',posesMDA(:,:,8,n),'Image9',posesMDA(:,:,9,n),'Image10',posesMDA(:,:,10,n),'Image11',posesMDA(:,:,11,n),'Image12',posesMDA(:,:,12,n),'Image13',posesMDA(:,:,13,n)));

end

Numtestpose=6;
xMDA=68;  
NumtestMDA=68;
count=0;

for n = 1:1:68
   trainingP1poseMDA(n*8-7,1) = struct('Label', n, 'Data', posesfinaleMDA(n).Image1.');
   trainingP1poseMDA(n*8-6,1) = struct('Label', n, 'Data', posesfinaleMDA(n).Image2.');
   trainingP1poseMDA(n*8-5,1) = struct('Label', n, 'Data', posesfinaleMDA(n).Image3.');
   trainingP1poseMDA(n*8-4,1) = struct('Label', n, 'Data', posesfinaleMDA(n).Image4.');
   trainingP1poseMDA(n*8-3,1) = struct('Label', n, 'Data', posesfinaleMDA(n).Image5.');
   trainingP1poseMDA(n*8-2,1) = struct('Label', n, 'Data', posesfinaleMDA(n).Image6.');
   trainingP1poseMDA(n*8-1,1) = struct('Label', n, 'Data', posesfinaleMDA(n).Image7.');
   trainingP1poseMDA(n*8,1) = struct('Label', n, 'Data', posesfinaleMDA(n).Image8.');
   count=count+1;
end
% disp(count);

i=1;
while i<=68
    testingP1poseMDA(5*i-4,1)=struct('Label',i,'Data',posesfinaleMDA(i).Image9.');
    testingP1poseMDA(5*i-3,1)=struct('Label',i,'Data',posesfinaleMDA(i).Image10.');
    testingP1poseMDA(5*i-2,1)=struct('Label',i,'Data',posesfinaleMDA(i).Image11.');
    testingP1poseMDA(5*i-1,1)=struct('Label',i,'Data',posesfinaleMDA(i).Image12.');
    testingP1poseMDA(5*i,1)=struct('Label',i,'Data',posesfinaleMDA(i).Image13.');
   i=i+1;
end

tablenew = struct2table(trainingP1poseMDA); 
sortedtable = sortrows(tablenew, 'Label'); 
trainingsortedP1MDA = table2struct(sortedtable); 

tablenewtest = struct2table(testingP1poseMDA); 
sortedtabletest = sortrows(tablenewtest, 'Label'); 
testingsortedP1MDA = table2struct(sortedtabletest); 

[meanMDApose,covarMDApose]=standardestimators(trainingP1poseMDA,xMDA*8,1);

[l,~]=size(covarMDApose(1).ClassCov);

[p,n]=size(covarMDApose);

for i=1:n
    lambdaMDA=2*ones(1,l);
    covarMDApose(i).ClassCov=covarMDApose(i).ClassCov + diag(lambdaMDA);
%     disp(det(covar(i).ClassCov));
end
%% Bayes Pose P1 trying

[BayesPCA,~,errorP1pose]=Bayes(testingP1poseMDA,meanMDApose,covarMDApose,Numtestpose);

%ERROR BAYES FINAL

bayestestingP1pose=sprintf('Percentage of error for Bayes Classifier with MDA for P1 pose is %f',errorP1pose);
disp(bayestestingP1pose);
%% KNN Pose P1 trying

k=3;
[~,testingarray,trainingarray,distancetruncated,indextruncated,labels,actuallabelspose,errorknnpose]=KNN(testingsortedP1MDA,trainingP1poseMDA,k,NumtestMDA*5,NumtestMDA*13);
knntestingpose=sprintf('Percentage of error for KNN Classifier with MDA for P1 pose is %f',errorknnpose);
disp(knntestingpose);



end


