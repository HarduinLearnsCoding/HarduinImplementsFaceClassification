function [Value] = rbfkernel(data1,data2,sigma)
   
    Value=exp(-norm(data1-data2)^2/sigma^2);
    
end

