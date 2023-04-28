function [   weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c,weight_forgetgate_x,weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h ]=LSTM_updata_weight_SGD(n,yita,Error,...
                                                   weight_input_x, weight_input_h, weight_inputgate_x,weight_inputgate_c,weight_forgetgate_x,weight_forgetgate_c,weight_outputgate_x,weight_outputgate_c,weight_preh_h,...
                                                   cell_state,h_state,input_gate,forget_gate,output_gate,gate,train_data,pre_h_state,input_gate_input, output_gate_input,forget_gate_input)

input_num=1000;  
cell_num=3;
output_num=300;
data_length=size(train_data,1);
data_num=size(train_data,2);
weight_preh_h_temp=weight_preh_h;

Op = struct;
Op.alpha = alpha;
b=0.01;
state = Op;
state_1 = Op;
state_2 = Op;
state_3 = Op;
state_4= Op;
state_5 = Op;
state_6 = Op;
state_7= Op;
state_8= Op;
state_9= Op;
stepsize=0.1;
p=2.5;





for m=1:output_num
    delta_weight_preh_h_temp(:,m)=2*Error(m,1)*pre_h_state;
end
%weight_preh_h_temp=weight_preh_h_temp-yita*delta_weight_preh_h_temp;
[updates_delta_weight_preh_h_temp,   state] = SGD(delta_weight_preh_h_temp,  state);
weight_preh_h_temp=weight_preh_h_temp-updates_delta_weight_preh_h_temp;

for num=1:output_num
    for m=1:data_length
        delta_weight_outputgate_x(m,:)=(2*weight_preh_h(:,num)*Error(num,1).*tanh(cell_state(:,n)))'.*exp(-output_gate_input).*(output_gate.^2)*train_data(m,n);
    end
    [updates_delta_weight_outputgate_x,   state_1] = SGD(delta_weight_outputgate_x,  state_1);
weight_outputgate_x=weight_outputgate_x-updates_delta_weight_outputgate_x;
    %weight_outputgate_x=weight_outputgate_x-yita*delta_weight_outputgate_x;
end

for num=1:output_num
for m=1:data_length
    delta_weight_inputgate_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*train_data(m,n);
end

 [updates_delta_weight_inputgate_x,   state_2] = SGD(delta_weight_inputgate_x,  state_2);
weight_inputgate_x=weight_inputgate_x-updates_delta_weight_inputgate_x;

%weight_inputgate_x=weight_inputgate_x-yita*delta_weight_inputgate_x;
end


if(n~=1)
    
    temp=train_data(:,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    for num=1:output_num
    for m=1:data_length
        delta_weight_input_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*train_data(m,n);
    end
    [updates_delta_weight_input_x,   state_3] = SGD(delta_weight_input_x, state_3);
    
    weight_input_x=weight_input_x-updates_delta_weight_input_x;
    end

    for num=1:output_num
    for m=1:data_length
        delta_weight_forgetgate_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2)*train_data(m,n);
    end
    
    [updates_delta_weight_forgetgate_x,   state_4] = SGD( delta_weight_forgetgate_x, state_4);
    
    weight_forgetgate_x=weight_forgetgate_x-updates_delta_weight_forgetgate_x;
    end
    %% ¸üĞÂweight_inputgate_c
    for num=1:output_num
    for m=1:cell_num
        delta_weight_inputgate_c(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*gate.*exp(-input_gate_input).*(input_gate.^2)*cell_state(m,n-1);
    end
    
    [updates_delta_weight_inputgate_c,   state_5] = SGD( delta_weight_inputgate_c,  state_5);
    
    weight_inputgate_c=weight_inputgate_c-updates_delta_weight_inputgate_c;
    end

    for num=1:output_num
    for m=1:cell_num
        delta_weight_forgetgate_c(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*cell_state(:,n-1)'.*exp(-forget_gate_input).*(forget_gate.^2)*cell_state(m,n-1);
    end
    
     [updates_delta_weight_forgetgate_c,   state_6] = SGD( delta_weight_forgetgate_c, state_6);
    
    weight_forgetgate_c=weight_forgetgate_c-updates_delta_weight_forgetgate_c;
    end

    for num=1:output_num
    for m=1:cell_num
        delta_weight_outputgate_c(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*tanh(cell_state(:,n))'.*exp(-output_gate_input).*(output_gate.^2)*cell_state(m,n-1);
    end
    
    [updates_delta_weight_forgetgate_c,   state_7] = SGD(delta_weight_outputgate_c,  state_7);
    
    weight_outputgate_c=weight_outputgate_c - updates_delta_weight_forgetgate_c;
    end
    
    temp=train_data(:,n)'*weight_input_x+h_state(:,n-1)'*weight_input_h;
    for num=1:output_num
    for m=1:output_num
        delta_weight_input_h(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*h_state(m,n-1);
    end
    
     [updates_delta_weight_input_h,   state_8] = SGD( delta_weight_input_h,  state_8);
    
    weight_input_h=weight_input_h-updates_delta_weight_input_h;
    end
else
  
    temp=train_data(:,n)'*weight_input_x;
    for num=1:output_num
    for m=1:data_length
        delta_weight_input_x(m,:)=2*(weight_preh_h(:,num)*Error(num,1))'.*output_gate.*(ones(size(cell_state(:,n)))-tanh(cell_state(:,n)).^2)'.*input_gate.*(ones(size(temp))-tanh(temp.^2))*train_data(m,n);
    end
    
    [updates_delta_weight_input_x,   state_9] = SGD( delta_weight_input_x,  state_9);
    weight_input_x=weight_input_x-updates_delta_weight_input_x;
    end
end
weight_preh_h=weight_preh_h_temp;

end