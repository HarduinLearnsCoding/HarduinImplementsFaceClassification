function [mean,covar]= standardestimators(trainingsorted,x,problem)
[l,~]=size(trainingsorted(1).Data);
switch problem
    case 1
        mean=[];
        covar=[];
        classsum=zeros(size(trainingsorted(1).Data));
        covarsum=zeros(l,l);
        initiallabel=1;
        for i=1:length(trainingsorted)
            if trainingsorted(i).Label==initiallabel
                classsum=trainingsorted(i).Data+classsum;
            else
                mean=[mean classsum/x];
                classsum=zeros(size(trainingsorted(1).Data));
                classsum=classsum+trainingsorted(i).Data;
                initiallabel=trainingsorted(i).Label;
            end
        end
        mean=[mean classsum/x].';
        covarlabel=1;
        for i=1:length(trainingsorted)
            if trainingsorted(i).Label==covarlabel
                covarsum=(trainingsorted(i).Data-mean(covarlabel,:).')*(trainingsorted(i).Data+mean(covarlabel,:).').'+covarsum;
            else
                covar=[covar struct('Label', covarlabel, 'ClassCov', covarsum/x)];
                covarsum=zeros(l,l);
                covarlabel=trainingsorted(i).Label;
                covarsum=covarsum+(trainingsorted(i).Data-mean(covarlabel,:).')*(trainingsorted(i).Data+mean(covarlabel,:).').';
            end
        end
        covar=[covar struct('Label', covarlabel, 'ClassCov', covarsum/x)];
           
    case 2
        class1sum=zeros(size(trainingsorted(1).Data));
        covar1sum=zeros(l,l);
        class2sum=zeros(size(trainingsorted(1).Data));
        covar2sum=zeros(l,l);
        class3sum=zeros(size(trainingsorted(1).Data));
        covar3sum=zeros(l,l);
        for i=1:length(trainingsorted)
            if trainingsorted(i).Label==1
                class1sum=trainingsorted(i).Data+class1sum;
            elseif trainingsorted(i).Label==2
                class2sum=trainingsorted(i).Data+class2sum;
            elseif trainingsorted(i).Label==3
                class3sum=trainingsorted(i).Data+class3sum;
            end
        end
        meanclass1=class1sum/x;
        meanclass2=class2sum/x;
        meanclass3=class3sum/x;
        mean=[meanclass1 meanclass2 meanclass3].';
        for i=1:length(trainingsorted)
            if trainingsorted(i).Label==1
                covar1sum=(trainingsorted(i).Data-meanclass1)*(trainingsorted(i).Data+meanclass1).'+covar1sum;
            elseif trainingsorted(i).Label==2
                covar2sum=(trainingsorted(i).Data-meanclass2)*(trainingsorted(i).Data+meanclass2).'+covar2sum;
            elseif trainingsorted(i).Label==3
                covar3sum=(trainingsorted(i).Data-meanclass3)*(trainingsorted(i).Data+meanclass3).'+covar3sum;
            end
        end
        covarclass1=covar1sum/x;
        covarclass2=covar2sum/x;
        covarclass3=covar3sum/x;
        covar=[];
        covar = [covar struct('Label', 1, 'ClassCov', covarclass1)];
        covar = [covar struct('Label', 2, 'ClassCov', covarclass2)];
        covar = [covar struct('Label', 3, 'ClassCov', covarclass3)];
end     
end