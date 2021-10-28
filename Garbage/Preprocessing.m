%% LOADING AND MAKING MATRICES
clc;
clear all;
close all;
load('data.mat');
load('illumination.mat');
load('pose.mat');
dimn=24*21;
faces=zeros(dimn,3,200);

for n=1:200
    faces(:,1,n)=reshape(face(:,:,3*n-2), [dimn,1]);
    faces(:,2,n)=reshape(face(:,:,3*n-1), [dimn,1]);
    faces(:,3,n)=reshape(face(:,:,3*n), [dimn,1]);
end

% disp(size(faces(:,1,:)))
% disp(size(faces(:,2,:)))
% disp(size(faces(:,3,:)))

dimn2=48*40;
% size(pose)

for n=1:68
    for j=1:13
        poses(:,j,n)=reshape(pose(:,:,j,n),[dimn2,1]);
    end
end

dimn3=dimn2;

for n=1:68
    for j=1:21
        illums(:,j,n)=reshape(reshape(illum(:,j,n), 48, 40),[dimn2,1]);
    end
end

% disp(size(faces))
% disp(size(poses))
% disp(size(illums))

%% CREATING TESTING TRAINING PARTITIONS

%function of data
%take split as input

split=1;
[m n l]=size(faces);
seq=1:1:200;
labeltr=horzcat(seq,seq);
labelte=[seq];
tedata=zeros(m,l);
trdata=zeros(m,n-1,l);
% disp(labeltr)

% disp([m n l])

switch split
    case 1
        trdata=horzcat(faces(:,1,:),faces(:,2,:));
        tedata=reshape(faces(:,3,:),[m,l]);
    case 2
        trdata=horzcat(faces(:,2,:),faces(:,3,:));
        tedata=reshape(faces(:,1,:),[m,l]);
    case 3
        trdata=horzcat(faces(:,1,:),faces(:,3,:));
        tedata=reshape(faces(:,2,:),[m,l]);
end

% disp(labelte)
% disp(size(trdata));
% disp(tedata)
%% MDA

%Priors?

% numofclasses=randi([1,200]);
% [m n l]=size(faces);
% meandata=zeros(m,1,numofclasses);
% for i=1:numofclasses
%     meandata(:,:,i)=reshape(mean(faces(:,:,i),2),[m,1]);
% end
% 
% covariancedata=zeros(m,m,numofclasses);
% for i=1:numofclasses
%     covariancedata(:,:,i)=cov(faces(:,:,i));
% end
% disp(covariancedata)
    

