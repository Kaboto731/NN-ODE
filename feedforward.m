function [a_H,z_H,a_out,z_out] = feedforward(w_H,b_H,w_out,x)
    % weighted inputs to hidden layer
    z_H = w_H*x + b_H;
    % activation of hidden layer
    a_H = sig(z_H);
    % weighted input to output layer
    z_out = sum(w_out.*a_H);
    % activation of output layer
    a_out = z_out;
end