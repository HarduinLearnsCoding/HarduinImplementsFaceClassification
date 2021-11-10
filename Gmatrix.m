function [matrixG] = Gmatrix(trainingSVM,sigma)

[col,~]=size(trainingSVM);
matrixG=zeros(col,col);

for i=1:col
    for j=1:col
        matrixG(i,j)=trainingSVM(j).Label * trainingSVM(i).Label * rbfkernel(trainingSVM(i).Data,trainingSVM(j).Data,sigma);
    end
end

end

