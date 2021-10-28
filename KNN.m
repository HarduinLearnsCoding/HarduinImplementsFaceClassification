function [classifiedKNN,testingarray,trainingarray,distance,index,labels,error]=KNN(testingsorted,trainingsorted,k,Numtest)
[l,z]=size(testingsorted(1).Data);
[m,~]=size(trainingsorted);
Numtrain=200-Numtest;
testingarray=zeros(Numtest*3,l);
prediction=0;
trainingarray=zeros(Numtrain*3,l);
distance=[];
index=[];
labels=[];
classifiedKNN=[];
count=0;
for i=1:Numtest*3
    testingarray(i,:)=testingsorted(i).Data.';
end
for j=1:Numtrain*3
    trainingarray(j,:)=trainingsorted(j).Data.';
end
for i=1:Numtest*3
    distance=[];
    for j=1:Numtrain*3
        distance=[distance; testingsorted(i).Label trainingsorted(j).Label abs(norm(testingarray(i)-trainingarray(j)))];
    end
    distance=sortrows(distance,3);
    distance=distance(1:k,:);
    index=mode(distance(:,2));
    classifiedKNN=[classifiedKNN struct('Predictedlabel',index,'ActualLabel',distance(1,1))];
    if index~=distance(1,1)
        count=count+1;
    end
end
error=(count/(Numtest*3))*100;

% for i=1:Numtest*3
%     distance=[];
%     for j=1:Numtrain*3
%         distance=[distance struct('TrainingLabel',trainingsorted(j).Label,'ActualLabel',testingsorted(i).Label,'Distance',abs(norm(testingsorted(i).Data)-norm(trainingsorted(j).Data)))];
%     end
%     tableKNN=struct2table(distance);
%     sortedtable = sortrows(tableKNN, 'Distance'); 
%     distancesorted = table2struct(sortedtable);
%     classifiedKNN=[classifiedKNN struct('PredictedLabel',mode(distancesorted.TrainingLabel),'ActualLabel',testingsorted(i).Label,'Data',testingsorted(i).Data)];
%end
% error=(count/Numtest)*100;
end




