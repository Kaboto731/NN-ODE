% Equation Parameters
% function
f = @(y,x) sin(x);
pf = @(y,x) 0;
IC = 0;
% exact solution
y = @(x) 1 - cos(x);

% Normal Distribution for weights
rng('default')

% Training Points
% number of points
n = 100;
% points
x = linspace(0,4*pi,n)';

% Network Parameters
% intial learning rate
eta = 0.01;
% drop rate
droprate = 1;
% hidden layer
% size
H = 10;
% biases
b_H = normrnd(0,1,[H,1]);
% weights
w_H = normrnd(0,1/sqrt(H),[H,1]);
% output layer
b_out = normrnd(0,1);
% weights
w_out = normrnd(0,1,[H,1]);

% Variables for Plotting Output of Network
% output layer
a_out = zeros(n,1);

% feedforward over batches
for i = 1:n
    [~,~,a_out(i),~] = feedforward(w_H,b_H,w_out,x(i));
end

% Plot Actual vs. ANN Solution
% figure
% hold on
% plot(x,y(x),'-k')
% plot(x,IC+x.*a_out,'-o')
% xlabel('x')
% ylabel('y')
% grid on
% title('Exact vs. ANN-initialized solution to y'' = y')
% legend('Exact','ANN','location','northwest')

% backpropagation algorithm
for i = 1:10000
    [w_H,b_H,w_out] = backpropagate(H,w_H,b_H,...
        w_out,n,x,f,pf,IC,eta,droprate,i);
end
% feedforward over training inputs
for i = 1:n
    [a_H,z_H,a_out(i),z_out] = feedforward(w_H,b_H,w_out,x(i));
end

% Plot Actual vs. ANN Solution
figure
hold on
plot(x,y(x),'-k')
plot(x,IC+x.*a_out,'-o')
xlabel('x')
ylabel('y')
grid on
title('Exact vs. ANN-computed solution to y'' = sin(x)')
legend('Exact','ANN','location','northwest')

% Error Plot
n_err = 100;
% sample
x_err = linspace(0,2*pi,n_err)';
a_out_err = zeros(n_err,1);
% feedforward over error-evaluating inputs
for i = 1:n_err
    [a_H,z_H,a_out_err(i),z_out] = feedforward(w_H,b_H,w_out,x_err(i));
end
% get errors
err = abs(y(x_err) - (IC + x_err.*a_out_err));
figure
plot(x_err,err,'--')
xlabel('x')
ylabel('error')
grid on
title('Absolute Error of ANN-computed solution to y'' = sin(x)')

% Extrapolation Plot
% m = 100;
% ex = linspace(0,10,m)';
% a_out = zeros(m,1);
% % feedforward over extrapolation points
% for i = 1:m
%     [a_H,z_H,a_out(i),z_out] = feedforward(w_H,b_H,w_out,ex(i));
% end
% figure
% hold on
% plot(ex,y(ex),'-k')
% plot(ex,IC+ex.*a_out,'-o')
% xlabel('x')
% ylabel('y')
% grid on
% title('Extrapolation of ANN-computed solution to y'' = y')
% legend('Exact','ANN','location','northwest')