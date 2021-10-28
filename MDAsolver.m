function[datavisualise,mean0,scatterbetween,scatterwithin,priors]=MDAsolver(facesfinale,mean,covar)
[m,n]=size(mean);
[y,z]=size(covar);
priors=1/m;
datavisualise=[];
mean0=zeros(size(mean(1,1)));
scatterwithin=zeros(size(covar(1).ClassCov));
scatterbetween=zeros(size(covar(1).ClassCov));
for i=1:y
    scatterwithin=scatterwithin+priors*covar(i).ClassCov;
end
%mu0 DONE
for i=1:m
    mean0=mean0+priors*mean(i,:);
end

for i=1:m
    scatterwithin=scatterwithin+(mean(i,:)-mean0).'*(mean(i,:)-mean0);
    datavisualise=[datavisualise struct('Eachiter', priors.*(mean(i,:)-mean0).'*(mean(i,:)-mean0))];
end
%scatterbetween 
end

