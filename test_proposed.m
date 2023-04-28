function [Error_iter]=test_proposed(data_length, data_num,train_data, test_data,test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
bias_output_gate,ab, weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c, weight_forgetgate_x,...
 weight_forgetgate_c, weight_outputgate_x, weight_outputgate_c,weight_preh_h, cost_gate,h_state, cell_state, yita)




m=7;  
gate=tanh(test_final'*weight_input_x+h_state(:,m-1)'*weight_input_h);
input_gate_input=test_final'*weight_inputgate_x+cell_state(:,m-1)'*weight_inputgate_c+bias_input_gate;
forget_gate_input=test_final'*weight_forgetgate_x+cell_state(:,m-1)'*weight_forgetgate_c+bias_forget_gate;
output_gate_input=test_final'*weight_outputgate_x+cell_state(:,m-1)'*weight_outputgate_c+bias_output_gate;
for n=1:cell_num
    input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
    forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
    output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
end
cell_state_test=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';
pre_h_state=tanh(cell_state_test').*output_gate;
test_h_state=(pre_h_state*weight_preh_h)';
Error_iter=mean(test_h_state~=test_data);
end