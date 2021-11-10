function[facesPCA,covarPCA,eigenvaluesPCA,eigenvectorsPCA,eigenvaluesdiagPCA,eigenvectorstrPCA]=PCAsolver(facestemp,numclasses)

newfacestemp=[];

for i=1:600
    newfacestemp(i,:)=reshape(facestemp(:,:,i),[504,1]);
end

covarPCA=cov(newfacestemp);
[eigenvectorsPCA,eigenvaluesPCA]=eig(covarPCA);
[eigenvaluesPCA, ind] = sort(eigenvaluesPCA,'descend');
eigenvectorsPCA = eigenvectorsPCA(:, ind);
eigenvaluesdiagPCA=diag(abs(eigenvaluesPCA));
eigenvaluesdiagPCA=eigenvaluesdiagPCA(1:numclasses-1,:);
eigenvectorstrPCA=eigenvectorsPCA(:,1:numclasses-1);
facesPCA=(newfacestemp*eigenvectorstrPCA);
end

