function [PredictedLabelSumAdaboost] = AdaSVMsolver(trainingSVM,testingSVM)
        
        filler=1;
        
        [coltrain,~]=size(trainingSVM);
            
        matrixG=Gmatrix(trainingSVM,filler,1,filler);
            
        f=ones(coltrain,1);

        for i=1:coltrain
                
              yvectorboost(i,1)=trainingSVM(i).Label;
                
        end

            Aeq=yvectorboost.';
            beq=0;
            lb=zeros(1,coltrain);
            A=[];
            b=[];
            x0=[];
            ub=[];

            options = optimset('Display', 'off');
            alphaboost = quadprog(matrixG,f,A,b,Aeq,beq,lb,ub,x0,options);
            
            [coltest,~]=size(testingSVM);
            PredictedLabelSumAdaboost=zeros(coltest,1);
    
            for i=1:coltest
                
                for j=1:coltrain
                    
                    PredictedLabelSumAdaboost(i,1)=PredictedLabelSumAdaboost(i,1)+alphaboost(j,1)*trainingSVM(j).Label*linearkernel(trainingSVM(j).Data,testingSVM(i).Data);
                
                end
                
            PredictedLabelSumAdaboost(i,1)=sign(PredictedLabelSumAdaboost(i,1));
            
            end
end

