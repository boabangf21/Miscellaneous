function [updates,   state] = Copy_of_AMSgrad(gradients, b, stepsize, p, state)
if nargin == 1
    state = struct;
end


 % p_1=0.4;
  
if ~isfield(state, 'p_1')
    state.p_1 = 0.01;
    %state.p_1 = 0.01;
end
if ~isfield(state, 'p_2')
    state.p_2 = 0.001;
    %state.p_1 = 0.01;
end
if ~isfield(state, 'step')
    state.step=  0.001;
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
    state.epsilon = 1e-8;
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
%v =  gradients.^2; 
% non-decreasing
%state.vhat = max(state.vhat, state.v);  % max H_t,i not enforced
%state.small =m(state.vhat, state.v);
%b=mean(state.v)

%
p_1=0.01;
p_new=0.4;

%     state.epsilon
for t=1:10
 


if  state.v < mean(gradients.^(2)) % threshold definition is important
% power_grad=(state.vhat + state.epsilon).^(p_1) .* log(state.vhat + state.epsilon);
 
  %state.b = state.beta3 * state.b + (1 - state.beta3) *  power_grad;
%state.bhat = max(state.bhat, state.b);

 %p_1 = p_1 -  state.step.*((((state.b))));% change
 % large catergory non-uniform case iteratively higher computation complexity
updates =  state.p_2 * state.m ./ ((state.v).^(.1));
else
    
   power_grad=(state.vhat + state.epsilon).^(p_1) .* log(state.vhat + state.epsilon);
 
  state.b = state.beta3 * state.b + (1 - state.beta3) *  power_grad;
%state.bhat = max(state.bhat, state.b);

 p_1 = p_1 -  state.step.*((((state.b))));% change 
    
    %updates = 0.01 *  state.m ./ ((state.vhat).^(p_1)); % small catergory uniform case use PADAM
   updates =  state.p_1 * state.m ./ ((state.v).^(p_1));
    
    %updates = state.p_1 *  state.m ./ ((state.vhat).^(0.05)); 
end

end







%+   state.epsilon

%p_1 = p_1(~isnan(p_1))
f = state.vhat;
%f=state.p_1;
% update parameters


% update iteration number
state.iteration = state.iteration + 1;

end