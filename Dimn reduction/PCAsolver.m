function[facesPCA,covarPCA,eigenvaluesPCA,eigenvectorsPCA,eigenvaluesdiagPCA,eigenvectorstrPCA]=PCAsolver(facestemp,numclasses)

covarPCA=cov(facestemp);
[eigenvectorsPCA,eigenvaluesPCA]=eig(covarPCA);
eigenvaluesdiagPCA=diag(abs(eigenvaluesPCA));
[eigenvaluesdiagPCA, ind] = sort(eigenvaluesdiagPCA,'descend');
eigenvectorsPCA = eigenvectorsPCA(:, ind);
eigenvaluesdiagPCA=eigenvaluesdiagPCA(1:numclasses-1,:);
eigenvectorstrPCA = eigenvectorsPCA(:,1:numclasses-1);
facesPCA=(facestemp*eigenvectorstrPCA);

end

