function [classifiedKNN,testingarray,trainingarray,distancetruncated,indextruncated,Labels,ActualLabels,error]=KNN(testingsorted,trainingsorted,k,Numtest,total)


[l,z]=size(testingsorted(1).Data);
[m,~]=size(trainingsorted);
Numtrain=total-Numtest;
testingarray=zeros(Numtest,l);
trainingarray=zeros(Numtrain,l);
distance=[];
index=[];
Labels=zeros(Numtest,k);
ActualLabels=zeros(Numtest,k);
classifiedKNN=[];
count=0;
distancetruncated=zeros(Numtest,k);
indextruncated=zeros(Numtest,k);
tempnew=0;



for i=1:Numtest

    testingarray(i,:)=testingsorted(i).Data.';

end

for j=1:Numtrain
    
    trainingarray(j,:)=trainingsorted(j).Data.';

end

for i=1:Numtest
    for j=1:Numtrain
        distance(i,j)=sqrt(sum((testingarray(i,:)-trainingarray(j,:)).^2));
    end
    [distance(i,:),index(i,:)]=sort(distance(i,:));
    distancetruncated(i,:)=distance(i,1:k);
    indextruncated(i,:)=index(i,1:k);
    for z=1:length(indextruncated(i,:))
        tempnew=indextruncated(i,z);
        Labels(i,z)=trainingsorted(tempnew).Label;
    end
    Labels(i,:)=mode(Labels(i,:));
    ActualLabels(i,:)=testingsorted(i).Label;
    
end
for i=1:Numtest
    if ActualLabels(i,1)~=Labels(i,1)
        count=count+1;
    end
end
error=(count/(Numtest))*100;

end




