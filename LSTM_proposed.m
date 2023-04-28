function [stats_proposed,Accuracy,Precision, Error_iter_proposed]=LSTM_proposed(data_length,data_num, train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
bias_output_gate,ab, weight_input_x, weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x, weight_inputgate_x_other, weight_inputgate_c,weight_inputgate_c_other, weight_forgetgate_x,weight_forgetgate_x_other,...
 weight_forgetgate_c,  weight_forgetgate_c_other, weight_outputgate_x, weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other, weight_preh_h, weight_preh_h_other,  cost_gate,h_state, cell_state, yita, tau); 

%h_state=rand(output_num,data_num);
%h_state=test_data;
%cell_state=rand(cell_num,data_num);

for iter=1:100
    %yita=0.01;
    for m=1:data_num
        
        if(m==1)
            gate=tanh(train_data(:,m)'*weight_input_x);
            input_gate_input=train_data(:,m)'*weight_inputgate_x+bias_input_gate;
            output_gate_input=train_data(:,m)'*weight_outputgate_x+bias_output_gate;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            forget_gate=zeros(1,cell_num);
            forget_gate_input=zeros(1,cell_num);
            cell_state(:,m)=(input_gate.*gate)';
        else
            gate=tanh(train_data(:,m)'*weight_input_x+h_state(:,m-1)'*weight_input_h);
            input_gate_input=train_data(:,m)'*weight_inputgate_x+cell_state(:,m-1)'*weight_inputgate_c+bias_input_gate;
            forget_gate_input=train_data(:,m)'*weight_forgetgate_x+cell_state(:,m-1)'*weight_forgetgate_c+bias_forget_gate;
            output_gate_input=train_data(:,m)'*weight_outputgate_x+cell_state(:,m-1)'*weight_outputgate_c+bias_output_gate;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            cell_state(:,m)=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';   
        end
        pre_h_state=tanh(cell_state(:,m)').*output_gate;
        % pre_h_state=(cell_state(:,m)').*output_gate;
        h_state(:,m)=(pre_h_state*weight_preh_h)';
        
        
        
        %ypred=softmax(h_state)
        %[model]=classify(train_data, h_state(1,:), [])
        %Ypred=classify(test_data, h_state);
        %tau=0.01;
        %r = So(tau,h_state);
        
        Error=(( h_state(:,m)- test_data(:,m)));
      
       %h_state_test(:,m)=((train_data(201:500,m)').*weight_preh_h)';
       %Error_iter=((h_state_test(:,m)-test_data(:,m)));
        
        
    stats_proposed= confusionmatStats(test_data(:,m), h_state(:,m));
        Error_Cost(1,iter)=sum(Error.^2);
        if(Error_Cost(1,iter)<cost_gate)
            flag=1;
            break;
        else
            [   weight_input_x,weight_input_x_other,...
                weight_input_h,weight_input_h_other,...
                weight_inputgate_x,weight_inputgate_x_other,...
                weight_inputgate_c,weight_inputgate_c_other,...
                weight_forgetgate_x,weight_forgetgate_x_other,...
                weight_forgetgate_c,weight_forgetgate_c_other,...
                weight_outputgate_x,weight_outputgate_x_other,...
                weight_outputgate_c,weight_outputgate_c_other,...
                weight_preh_h, weight_preh_h_other, tau]=LSTM_updata_weight_proposed(m,yita,Error,...
                                                   weight_input_x,weight_input_x_other,...
                                                   weight_input_h,weight_input_h_other,...
                                                   weight_inputgate_x, weight_inputgate_x_other,...
                                                   weight_inputgate_c,weight_inputgate_c_other,...
                                                   weight_forgetgate_x,weight_forgetgate_x_other,...
                                                   weight_forgetgate_c, weight_forgetgate_c_other,...
                                                   weight_outputgate_x,weight_outputgate_x_other,...
                                                   weight_outputgate_c,weight_outputgate_c_other,...
                                                   weight_preh_h,weight_preh_h_other,...
                                                   cell_state,h_state,...
                                                   input_gate,forget_gate,...
                                                   output_gate,gate,...
                                                   train_data,pre_h_state,...
                                                   input_gate_input,...
                                                   output_gate_input,...
                                                   forget_gate_input, tau);

        end
   % Error_iter(:, iter) = mean(all(h_state(:,m) - test_data(:,m),2))
    end
    if(Error_Cost(1,iter)<cost_gate)
        break;
    end


s=6;
            gate=tanh(test_final(:,s)'*weight_input_x+h_state(:,s-1)'*weight_input_h);
            input_gate_input=test_final(:,s)'*weight_inputgate_x+cell_state(:,s-1)'*weight_inputgate_c+bias_input_gate;
            forget_gate_input=test_final(:,s)'*weight_forgetgate_x+cell_state(:,s-1)'*weight_forgetgate_c+bias_forget_gate;
            output_gate_input=test_final(:,s)'*weight_outputgate_x+cell_state(:,s-1)'*weight_outputgate_c+bias_output_gate;
            for n=1:cell_num
                input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
                output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
            end
            cell_state_test(:,s)=(input_gate.*gate+cell_state(:,s-1)'.*forget_gate)';   
       
        %pre_h_state=tanh(cell_state_test(:,s)').*output_gate;
       pre_h_state=tanh(cell_state_test(:,s)').*output_gate;
        %h_state_test(:,s)=round(pre_h_state*weight_preh_h)';
        h_state_test(:,s)=(pre_h_state*weight_preh_h)';
        
        
        %ypred=softmax(h_state)
        %[model]=classify(train_data, h_state(1,:), [])
        %Ypred=classify(test_data, h_state);
        %tau=0.01;
        %r = So(tau,h_state);
        
        Error_test=((h_state_test(:,s) -  test_hstate(:,s)));
%Error_test=((h_state_test(:,s) -  test_hstate(:,s)));
[field2, field3]  = confusionmatStats(test_hstate(:,s)', h_state_test(:,s)');

Accuracy(:, iter)=mean(field2);
Precision(:, iter)=mean(field3);
 % Error_iter_proposed(:, iter)=mean(all( Error_test, 2))
Error_iter_proposed(:, iter)=abs(mean( Error));
end
end

