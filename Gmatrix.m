function [matrixG] = Gmatrix(trainingSVM,sigma,kerneltype,r)

[col,~]=size(trainingSVM);
matrixG=zeros(col,col);

switch kerneltype
    case 2
        for i=1:col
            for j=1:col
                matrixG(i,j)=trainingSVM(j).Label * trainingSVM(i).Label * rbfkernel(trainingSVM(i).Data,trainingSVM(j).Data,sigma);
            end
        end
    case 1
        for i=1:col
            for j=1:col
                matrixG(i,j)=real(trainingSVM(j).Label * trainingSVM(i).Label * linearkernel(trainingSVM(i).Data,trainingSVM(j).Data));
            end
        end
    case 3
         for i=1:col
            for j=1:col
                matrixG(i,j)=trainingSVM(j).Label * trainingSVM(i).Label * polynomialkernel(trainingSVM(i).Data,trainingSVM(j).Data,r);
            end
         end
end

end

