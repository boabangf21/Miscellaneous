function [updates,   state] = AMSgrad_proposed(gradients, b, stepsize, p, state)
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

state.vhat = state.beta2 * state.vhat + (1 - state.beta2) * (gradients).^(2);
b=state.vhat;
%state.vhat = gradients.^(2)
%state.vhat = max(state.vhat, state.v);
%v =  state.vhat
% non-decreasing

%state.vhat = max(state.vhat, state.v);  % max H_t,i not enforced

%updates = 0.01.* state.m ./ ((state.vhat).^(0.1));
%state.small =m(state.vhat, state.v);
%b=mean(state.v)
%a_min=numel((sign((mean((state.vhat)/2)-(state.vhat))+ 1)./2) -   nnz(sign((mean((state.vhat)/2)-(state.vhat))+ 1)./2;
%a_max=numel((sign((state.vhat)-(mean((state.vhat)/2)))+1)./2)-nnz(sign((state.vhat)-(mean((state.vhat)/2))+1)./2;

   
updates1 = 0.003.* state.m .*(((sign(((mean((state.vhat))/0.01))-(state.vhat)) + 1 )))./ (2.* (state.vhat).^0.25);
updates2 =  0.01.* state.m .*(((sign(((state.vhat)-(mean((state.vhat))/0.01)))) + 1 ))./ ((2.* (state.vhat).^0.25));


updates = updates2 + updates1;
 % sum update rule
%c=(sign((state.vhat)-(mean((state.vhat))/0.1)) + 1 )./ 2
%+ state.epsilon.*state.vhat
  %if state.v < 1e-6 % needle_passing settings
%J=(sign((state.vhat)-mean(state.vhat)) + 1 )./2
%updates =  0.001 * state.m ./ ((state.vhat).^(0.2) );
% update iteration number
state.iteration = state.iteration + 1;

end