function [   weight_input_x,weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x, weight_inputgate_x_other, weight_inputgate_c, weight_inputgate_c_other, weight_forgetgate_x, weight_forgetgate_x_other, weight_forgetgate_c,weight_forgetgate_c_other, weight_outputgate_x, weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other, weight_preh_h,  weight_preh_h_other, tau]=LSTM_updata_weight_proposed_signed(n,yita,Error,...
                                                   weight_input_x, weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x,weight_inputgate_x_other, weight_inputgate_c, weight_inputgate_c_other, weight_forgetgate_x, weight_forgetgate_x_other, weight_forgetgate_c,  weight_forgetgate_c_other, weight_outputgate_x,  weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other,  weight_preh_h, weight_preh_h_other,...
                                                   cell_state,h_state,input_gate,forget_gate,output_gate,gate,train_data,pre_h_state,input_gate_input, output_gate_input,forget_gate_input, tau)
input_num=180;  
cell_num=1;
output_num=20;      
data_length=size(train_data,1);
data_num=size(train_data,2);
weight_preh_h_temp=weight_preh_h;

Op = struct;
Op.alpha = yita;
b=0.01;
%state = Op;
state = Op;
state_other = Op;
state_tau = Op;
state_1 = Op;
state_1_other = Op;
state_1_tau = Op;
state_2 = Op;
state_2_other = Op;
state_2_tau = Op;
state_3 = Op;
state_3_other = Op;
state_3_tau = Op;
state_4= Op;
state_4_other = Op;
state_4_tau = Op;
state_5 = Op;
state_5_other = Op;
state_5_tau = Op;
state_6 = Op;
state_6_other = Op;
state_6_tau = Op;
state_7= Op;
state_7_other = Op;
state_7_tau = Op;
state_8= Op;
state_8_other = Op;
state_8_tau = Op;
state_9= Op;
state_9_other = Op;
state_9_tau = Op;

stepsize=0.1;
p=2.5;





for m=1:output_num
    delta_weight_preh_h_temp(:,m)=2*Error(m,1)*pre_h_state;
end
%weight_preh_h_temp=weight_preh_h_temp-yita*delta_weight_preh_h_temp;
[updates_delta_weight_preh_h_temp,   state] = AMSgrad_signed(delta_weight_preh_h_temp, b, stepsize, p, state);
weight_preh_h_temp=weight_preh_h_temp-updates_delta_weight_preh_h_temp;

for num=1:output_num
    for m=1:data_length
        delta_weight_outputgate_x(m,:)=(2*weight_preh_h(:,num)*Error(num,1).*tanh(cell_state(:,n)))'.*exp(-output_gate_input).*(output_gate.^2)*train_data(m,n)-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        delta_weight_outputgate_x_other(m,:)= tau.*sign(weight_preh_h_other(:,num)') + tau.*(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        tau_gradient_outputgate_x =  (2*weight_preh_h(:,num)*Error(num,1).*tanh(cell_state(:,n)))'.*exp(-output_gate_input).*(output_gate.^2)*train_data(m,n)-sign(weight_preh_h_other(:,num)')-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')-(1/2)*(abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')).^2;
    end
     %[updates_tau_gradient,   state_1] = Copy_of_AMSgrad(tau_gradient, b, stepsize, p, state_1_tau);
    
    
    [updates_delta_weight_outputgate_x,   state_9] = AMSgrad_signed(delta_weight_outputgate_x, b, stepsize, p, state_9);
    [updates_delta_weight_outputgate_x_other,   state_9_other] = AMSgrad_signed(delta_weight_outputgate_x_other, b, stepsize, p, state_9_other);
    % tau =tau - 0.0001*tau_gradient_outputgate_x;
    weight_outputgate_x=weight_outputgate_x-updates_delta_weight_outputgate_x;
     weight_outputgate_x = So(tau,weight_outputgate_x);
     weight_outputgate_x_other=weight_outputgate_x_other - updates_delta_weight_outputgate_x_other;
     weight_outputgate_x_other = So(tau,weight_outputgate_x_other);
   


end

for num=1:output_num
for m=1:data_length
    delta_weight_inputgate_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*train_data(m,n)-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
delta_weight_inputgate_x_other(m,:)= tau.*sign(weight_preh_h_other(:,num)') + tau.*(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        tau_gradient_inputgate_x =  (weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*train_data(m,n);-sign(weight_preh_h_other(:,num)')-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')-(1/2)*(abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')).^2;
 %delta_weight_inputgate_x_other(m,:)=tau.*sign(weight_preh_h_other(:,num)') + tau.*(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
%tau_gradient_inputgate_x= 2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*train_data(m,n);


end
 %tau =tau - 0.001*tau_gradient_inputgate_x;
 [updates_delta_weight_inputgate_x,   state_1] = AMSgrad_signed(delta_weight_inputgate_x, b, stepsize, p, state_1);
weight_inputgate_x=weight_inputgate_x-updates_delta_weight_inputgate_x;
[updates_delta_weight_inputgate_x_other,   state_1_other] = AMSgrad_signed(delta_weight_inputgate_x_other, b, stepsize, p, state_1_other);
    %weight_inputgate_x=weight_inputgate_x-updates_delta_weight_inputgate_x;
     weight_inputgate_x = So(tau,weight_inputgate_x);
     weight_inputgate_x_other=weight_inputgate_x_other - updates_delta_weight_inputgate_x_other;
     weight_inputgate_x_other = So(tau,weight_inputgate_x_other);
   
%weight_inputgate_x=weight_inputgate_x-yita*delta_weight_inputgate_x;
end


if(n~=1)
    
    temp=train_data(:,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    for num=1:output_num
    for m=1:data_length
        delta_weight_input_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*train_data(m,n)-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        delta_weight_input_x_other(m,:)= tau.*sign(weight_preh_h_other(:,num)') + tau.*(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        tau_gradient_input_x =  2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*train_data(m,n)-sign(weight_preh_h_other(:,num)')-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')-(1/2)*(abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')).^2;
    end
   
    [updates_delta_weight_input_x,   state_2] = AMSgrad_signed(delta_weight_input_x, b, stepsize, p, state_2);
    weight_input_x=weight_input_x-updates_delta_weight_input_x;
    
    % tau =tau - 0.001*tau_gradient_input_x;
    [updates_delta_weight_input_x_other,   state_2_other] = AMSgrad_signed(delta_weight_input_x_other, b, stepsize, p, state_2_other);
    %weight_input_x=weight_input_x-updates_delta_weight_input_x;
     weight_input_x = So(tau,weight_input_x);
     weight_input_x_other=weight_input_x_other - updates_delta_weight_input_x_other;
     weight_input_x_other = So(tau,weight_input_x_other);
   
    
    
    end

    for num=1:output_num
    for m=1:data_length
        delta_weight_forgetgate_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2)*train_data(m,n)-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        delta_weight_forgetgate_x_other(m,:)= tau.*sign(weight_preh_h_other(:,num)') + tau.*(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        tau_gradient_forgetgate_x = 2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2)*train_data(m,n)-sign(weight_preh_h_other(:,num)')-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')-(1/2)*(abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')).^2; 
    end
    
    [updates_delta_weight_forgetgate_x,   state_3] = AMSgrad_signed( delta_weight_forgetgate_x, b, stepsize, p, state_3);
    weight_forgetgate_x=weight_forgetgate_x-updates_delta_weight_forgetgate_x;
    
    %tau =tau - 0.001*tau_gradient_forgetgate_x;
    [updates_delta_weight_forgetgate_x_other,   state_3_other] = AMSgrad_signed(delta_weight_forgetgate_x_other, b, stepsize, p, state_3_other);
   % weight_forgetgate_x=weight_forgetgate_x-updates_delta_weight_forgetgate_x;
     weight_forgetgate_x = So(tau,weight_forgetgate_x);
     weight_forgetgate_x_other=weight_forgetgate_x_other - updates_delta_weight_forgetgate_x_other;
     weight_forgetgate_x_other = So(tau,weight_forgetgate_x_other);
   
    
    
    
    
    end
    %% ¸üÐÂweight_inputgate_c
    for num=1:output_num
    for m=1:cell_num
        delta_weight_inputgate_c(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*cell_state(m,n-1)-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        delta_weight_inputgate_c_other(m,:)= tau.*sign(weight_preh_h_other(:,num)') + tau.*(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        tau_gradient_inputgate_c =  2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*cell_state(m,n-1)-sign(weight_preh_h_other(:,num)')-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')-(1/2)*(abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')).^2; 
    
    end
    
    [updates_delta_weight_inputgate_c,   state_4] = AMSgrad_signed( delta_weight_inputgate_c, b, stepsize, p, state_4);
    weight_inputgate_c=weight_inputgate_c-updates_delta_weight_inputgate_c;
    %tau =tau - 0.1*tau_gradient_inputgate_c;
    [updates_delta_weight_inputgate_c_other,   state_4_other] = AMSgrad_signed(delta_weight_inputgate_c_other, b, stepsize, p, state_4_other);
    %weight_inputgate_c=weight_inputgate_c-updates_delta_weight_inputgate_c;
     weight_inputgate_c = So(tau,weight_inputgate_c);
     weight_inputgate_c_other=weight_inputgate_c_other - updates_delta_weight_inputgate_c_other;
     weight_inputgate_c_other = So(tau,weight_inputgate_c_other);
   
    
    
    
    end

    for num=1:output_num
    for m=1:cell_num
        delta_weight_forgetgate_c(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2)*cell_state(m,n-1)-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        delta_weight_forgetgate_c_other(m,:)= tau.*sign(weight_preh_h_other(:,num)') + tau.*(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        tau_gradient_forgetgate_c = 2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2)*cell_state(m,n-1) -sign(weight_preh_h_other(:,num)')-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')-(1/2)*(abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')).^2; 
    end
    [updates_delta_weight_forgetgate_c,   state_5] = AMSgrad_signed( delta_weight_forgetgate_c, b, stepsize, p, state_5);
    weight_forgetgate_c=weight_forgetgate_c-updates_delta_weight_forgetgate_c;
    
    %tau =tau - 0.1*tau_gradient_forgetgate_c;
    [updates_delta_weight_forgetgate_c_other,   state_5_other] = AMSgrad_signed(delta_weight_forgetgate_c_other, b, stepsize, p, state_5_other);
    %weight_forgetgate_c=weight_forgetgate_c-updates_delta_weight_forgetgate_c;
     weight_forgetgate_c = So(tau,weight_forgetgate_c);
     weight_forgetgate_c_other=weight_forgetgate_c_other - updates_delta_weight_forgetgate_c_other;
     weight_forgetgate_c_other = So(tau,weight_forgetgate_c_other);
   
    
    
    end

    for num=1:output_num
    for m=1:cell_num
        delta_weight_outputgate_c(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*tanh(cell_state(:,n))'.*exp(-output_gate_input).*(output_gate.^2)*cell_state(m,n-1)-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        delta_weight_outputgate_c_other(m,:)= tau.*sign(weight_preh_h_other(:,num)') + tau.*(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        tau_gradient_outputgate_c =  2*(weight_preh_h(:,num)*Error(num,1))'.*tanh(cell_state(:,n))'.*exp(-output_gate_input).*(output_gate.^2)*cell_state(m,n-1)-sign(weight_preh_h_other(:,num)')-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')-(1/2)*(abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')).^2; 
    end
    
    [updates_delta_weight_ouputgate_c,   state_6] = AMSgrad_signed(delta_weight_outputgate_c, b, stepsize, p, state_6);
    weight_outputgate_c=weight_outputgate_c - updates_delta_weight_ouputgate_c;
    
    %tau =tau - 0.001*tau_gradient_outputgate_c;
    [updates_delta_weight_outputgate_c_other,   state_6_other] = AMSgrad_signed(delta_weight_outputgate_c_other, b, stepsize, p, state_6_other);
    %weight_outputgate_c=weight_outputgate_c-updates_delta_weight_outputgate_c_other;
     weight_outputgate_c = So(tau,weight_outputgate_c);
     weight_outputgate_c_other=weight_outputgate_c_other - updates_delta_weight_outputgate_c_other;
     weight_outputgate_c_other = So(tau,weight_outputgate_c_other);
    
    end
    
    temp=train_data(:,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    for num=1:output_num
    for m=1:output_num
        delta_weight_input_h(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*h_state(m,n-1)-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
       delta_weight_input_h_other(m,:)= tau.*sign(weight_preh_h_other(:,num)') + tau.*(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        tau_gradient_input_h =  2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*h_state(m,n-1)-sign(weight_preh_h_other(:,num)')-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')-(1/2)*(abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')).^2; 
    end
    
     [updates_delta_weight_input_h,   state_7] = AMSgrad_signed( delta_weight_input_h, b, stepsize, p, state_7);
    weight_input_h=weight_input_h-updates_delta_weight_input_h;
   
    %tau =tau - 0.1*tau_gradient_input_h;
    [updates_delta_weight_input_h_other,   state_7_other] = AMSgrad_signed(delta_weight_input_h_other, b, stepsize, p, state_7_other);
    %weight_input_h=weight_input_h-updates_delta_weight_input_h;
     weight_input_h = So(tau,weight_input_h);
     weight_input_h_other=weight_input_h_other - updates_delta_weight_input_h_other;
     weight_input_h_other = So(tau,weight_input_h_other);
    
    
    
    
    end
else
  
    temp=train_data(:,n)'*weight_input_x;
    for num=1:output_num
    for m=1:data_length
        delta_weight_input_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*train_data(m,n)-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
       delta_weight_input_x_other(m,:)= tau.*sign(weight_preh_h_other(:,num)') + tau.*(weight_preh_h(:,num)' - weight_preh_h_other(:,num)');
        tau_gradient_input_x =  2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*train_data(m,n)-sign(weight_preh_h_other(:,num)')-tau.*abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')-(1/2)*(abs(weight_preh_h(:,num)' - weight_preh_h_other(:,num)')).^2; 
    end
    
    [updates_delta_weight_input_x,   state_8] = AMSgrad_signed( delta_weight_input_x, b, stepsize, p, state_8);
    weight_input_x=weight_input_x-updates_delta_weight_input_x;
    
    %tau =tau - 0.1*tau_gradient_input_x;
    [updates_delta_weight_input_x_other,   state_8_other] = AMSgrad_signed(delta_weight_input_x_other, b, stepsize, p, state_8_other);
    %weight_input_h=weight_input_h-updates_delta_weight_input_h;
     weight_input_x = So(tau,weight_input_x);
     weight_input_x_other=weight_input_x_other - updates_delta_weight_input_x_other;
     weight_input_x_other = So(tau,weight_input_x_other);
    
    
    end
end
weight_preh_h=weight_preh_h_temp;

end