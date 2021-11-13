function [covarfixed]=regularise(covar,l)
lambda=zeros(1,l);
lambda=diag(lambda);
covarfixed=covar;
for i=1:3
    while det(covarfixed(i).ClassCov)==0
        lambda=lambda+diag(.1*ones(1,l));
        covarfixed=[ covarfixed struct('Label', covarfixed(i).Label, 'ClassCov', covarfixed(i).ClassCov+lambda)];
    end
    disp(det(covarfixed(i).ClassCov));
end

        
        
%NOT BEING USED RIGHT NOW










end
