function [PredictedLabelSum,errorSVM] = KfoldKernelSVMsolver(trainingSVM,testingSVM,kerneltype,sigma,r)

[coltrain,~]=size(trainingSVM);
matrixG=Gmatrix(trainingSVM,sigma,kerneltype,r);
matrixG=real(matrixG);
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
end

