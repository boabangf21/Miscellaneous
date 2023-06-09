function [updates, state] = Adam2(gradients, b, stepsize, p,  state)
%ADAM2 Summary of this function goes here
%   Detailed explanation goes here

if nargin == 1
    state = struct;
end

if ~isfield(state, 'beta1')
    state.beta1 = 0.9;
end
if ~isfield(state, 'beta2')
    state.beta2 = 0.99;
end
if ~isfield(state, 'beta1t')
    state.beta1t = state.beta1;
end
if ~isfield(state, 'beta2t')
    state.beta2t = state.beta2;
end
if ~isfield(state, 'epsilon')
    state.epsilon = 1e-15;
end
if ~isfield(state, 'm')
    state.m = zeros(size(gradients));
end
if ~isfield(state, 'v')
    state.v = zeros(size(gradients));
end
if ~isfield(state, 'vhat')
    state.vhat = state.v;
end
if ~isfield(state, 'alpha')
    state.alpha = 1e-2;
end

% update biased first moment estimate
state.m = state.beta1 * state.m + (1 - state.beta1) * gradients;

% update biased second raw moment estimate
state.v = state.beta2 * state.v + (1 - state.beta2) * gradients.^2;

% bias correction
bc = sqrt(1 - state.beta2t) / (1 - state.beta1t);

% compute bias-corrected second raw moment estimate
state.vhat = max(state.vhat, state.v);

% update parameters
updates = 0.00003 * state.m * bc ./ ((state.vhat).^0.5 );
state.beta1t = state.beta1 * state.beta1t;
state.beta2t = state.beta2 * state.beta2t;


end

