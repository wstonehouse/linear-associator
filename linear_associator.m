%{
Will Stonehouse
Psych 186B: Neural Networks
Homework 5: Federation Intelligence Exercise
Assignment: Train a linear associator to identify the origins of ships 
based on sensor scans, so that the Enterprise will be able to take 
appropriate action when another ship approaches.
%}

%% Vars

% x_set : training input vectors
% y_set : training labels

% A : connectivity matrix
% count : amount of associations

%% Loading data

clear;
close all;
clc;

% data with labels
y_set = table2array(readtable("train.xlsx"));
x_set = y_set;

% data without labels
x_set(5:11,:) = 0;

%% Create connectivity matrix

[dim, count] = size(x_set);

% Set of associations
A_set=zeros(dim,dim,count);

for i=1:count
    x = x_set(:,i);
    y = y_set(:,i);
    A = x*y.';
    A_set(:,:,i) = A;
end
A = sum(A_set,3);


%% Train neural network

A = zeros(11);
iterations = 1000;

% Store the error^2 to track error
e = [];
for i=1:iterations
    % Choose a random i/o pair
    rand = randi(count,1);
    x = x_set(:,rand);
    y = y_set(:,rand);
    % Compute k (learning constant)
    k = (1/(x.'*x)-0.0001)/i;
    % Actual output vector g
    g = A*x;
    % Compute error vector
    error = (y-g);
    % Delta A 
    delta = k*error*x.';
    % Add both A and delta A
    %A=A+delta;
    A = A+delta+y*x.';
end

% Test the associations with cosine
% disp(cos(A,x_set,y_set));

%% Testing neural network

% the unlabeled test data
testData = table2array(readtable('test.xlsx'));

% the average feature for each type of ship
avg = table2array(readtable("ship_avg.xlsx"));
klingon = avg(:,1);
romulan = avg(:,2);
antarean = avg(:,3);
federation = avg(:,4);

for i=1:20
    g = testData(:,i);
    g_p = A*g;
    test = g_p./iterations; % divide each element by the amount of iterations

    % Calculate Mean Squared Error for each ship type
    diff = zeros(4,1);
    diff(1) = (sum((test - klingon).^2))/iterations;
    diff(2) = (sum((test - romulan).^2))/iterations;
    diff(3) = (sum((test - antarean).^2))/iterations;
    diff(4) = (sum((test - federation).^2))/iterations;
    lowest = diff(1);
    % Which MSE is the lowest?
    for j = 1:4
        if(lowest>diff(j,1))
            lowest = diff(j,1);
        end
    end
    disp(i);
    if(lowest == diff(1))
        disp('Klingon');
        disp('Hostile');
    elseif(lowest == diff(2))
        disp('Romulan');
        disp('Alert!');
    elseif(lowest == diff(3))
        disp('Antarean');
        disp('Friendly');
    elseif(lowest == diff(4))
        disp('Federation');
        disp('Friendly');
    end
end

%% Functions

function [cos] = cos(A,x_set,y_set)
    dim=size(x_set);
    cos=[];
    for i=1:dim(2)
        y=y_set(:,i);
        x=x_set(:,i);
        g=A*x;
        cos(end+1)=dot(y,g);
    end
    cos=mean(cos);
end