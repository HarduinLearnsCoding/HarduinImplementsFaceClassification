function [Value] = polynomialkernel(data1,data2,r)

    Value=(data1.'*data2+1)^r;
    
end

