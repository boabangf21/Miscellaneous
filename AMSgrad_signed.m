function [updates,   state] = AMSgrad_signed(gradients, b, stepsize, p, state)
if nargin == 1
    state = struct;
end


 % p_1=0.4;
  
if ~isfield(state, 'p_1')
    state.p_1 = p;
end

if ~isfield(state, 'beta1')
    state.beta1 = 0.9;
end

if ~isfield(state, 'beta2') 
    state.beta2 = 0.99;
end
if ~isfield(state, 'beta3') 
    state.beta3 = 0.9;
end
if ~isfield(state, 'epsilon')
    state.epsilon = 1e-15;
   % state.epsilon = 1e-10;
end
if ~isfield(state, 'iteration')
    state.iteration = 1;
end
if ~isfield(state, 'm')
    state.m = zeros(size(gradients));
end

if ~isfield(state, 'b')
    state.b = zeros(size(gradients));
end

if ~isfield(state, 'v')
    state.v = zeros(size(gradients));
end
if ~isfield(state, 'vhat')
    state.vhat = zeros(size(gradients));
end
if ~isfield(state, 'alpha')
    state.alpha = 1e-2;
end

% update biased first moment estimate
state.m = state.beta1 * state.m + (1 - state.beta1) * gradients;
 %state.m = gradients; 
% update biased second raw moment estimate

state.v = state.beta2 * state.v + (1 - state.beta2) * gradients.^(2);
%state.v = state.v + gradients.^2; 
% non-decreasing
state.vhat = max(state.vhat, state.v);
%state.vhat = state.v;
%
p_1=0.1;
p_new=0.1;

%     state.epsilon
%for t=1:10
 power_grad=(state.v +  state.epsilon ).^(p_1) .* log(state.v +  state.epsilon );
 %state.b = state.beta3 * state.b + (1 - state.beta3) *  power_grad;
%p_1 = p_1 - 0.04.*sign((power_grad));
f=mean(p_1);
%p_1 = p_1 - 0.05.*sign(0.1*sum(power_grad));
p_1 = p_1 - 0.08.*sign(((power_grad)));

%p_1 = p_1 - 0.05.*sign(((power_grad))).*abs(2.^round(log2(sqrt(((power_grad))))));

%p_1 = p_1 - 0.038.*((sign((power_grad))))
%p_new =max(p_new, p_1);
%p_1 = p_1 - 0.01.*((Error))

%updates = 0.01* state.m ./ ((state.v  ).^( p_new));
%end

%updates = 0.0001 * state.m ./ ((state.vhat).^(0.4));
updates =   0.003.* state.m ./ ((state.vhat).^(0.4));

 %updates = 0.1 * state.m ./ ((state.vhat).^(0.05));


%+   state.epsilon

%p_1 = p_1(~isnan(p_1))
f = state.vhat;
%f=state.p_1;
% update parameters


% update iteration number
state.iteration = state.iteration + 1;

end