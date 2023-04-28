%clear;
%clc;
%X=rand(100,76);
%X_2_new=abs(load('Needle_Passing_I001.txt'));
%X_2_new=abs(load('Suturing_B001.txt'));
%X_2_new=abs(load('Knot_Tying_B002.txt'));
X_2=X_2_new(501:700,1:76);
n=size(X_2,1);
X_1_new=abs(load('Suturing_B002.txt')); 
%X_1_new=abs(load('Knot_Tying_B001.txt')); 

X_1=X_1_new(501:700,1:76);      

data_length=size(X_1,1);
 
X=X_1_new(501:700,1:76); 

%train_data=[X_1(1:100   0,1:20)];
%test_final=[X_2(1:1000,1:20)];

%test_data_1=abs(X_1(1:1200,1:20));
%test_data=(test_data_1(901:1000,1:20));
%[train_data,test_data]=LSTM_data_process()
%test_hstate=[X_2(1:100,1:76)];
%test_hstate=(test_data_1(1:100,1:20));




ntrain = ceil(n * 0.9); % number of training data;
ntrain_2 = ceil(n * 0.9); % number of training data;
idx =randperm(n);
idx_2 =randperm(n);
X_1 = X_1(idx, :);
X_2 = X_2(idx, :);
%Total = Total(idx, :);
%Total_2 = Total(idx_2, :);
train_data = X_1(1: ntrain, :);
%ytrain = y(1: ntrain);
test_data= X(ntrain + 1:end, :);
%test_final= X(ntrain + 1:end, :);
test_hstate = X(ntrain + 1:end, :);
%test_data= X(1: ntrain, :);
test_final= X_2(1: ntrain, :);



data_num=size(train_data,2);
data_length=size(train_data,1);
input_num=180;  
cell_num=1;
output_num=20;  



bias_input_gate=rand(1,cell_num);
bias_forget_gate=rand(1,cell_num);
bias_output_gate=rand(1,cell_num);



ab=25;
weight_input_x=rand(input_num,cell_num)/ab;
weight_input_x_other=rand(input_num,cell_num)/ab;

weight_input_h=rand(output_num,cell_num)/ab;
weight_input_h_other=rand(output_num,cell_num)/ab;

weight_inputgate_x=rand(input_num,cell_num)/ab;
weight_inputgate_x_other=rand(input_num,cell_num)/ab;

weight_inputgate_c=rand(cell_num,cell_num)/ab;
weight_inputgate_c_other=rand(cell_num,cell_num)/ab;

weight_forgetgate_x=rand(input_num,cell_num)/ab;
weight_forgetgate_x_other=rand(input_num,cell_num)/ab;

weight_forgetgate_c=rand(cell_num,cell_num)/ab;
weight_forgetgate_c_other=rand(cell_num,cell_num)/ab;

weight_outputgate_x=rand(input_num,cell_num)/ab;
weight_outputgate_x_other=rand(input_num,cell_num)/ab;


weight_outputgate_c=rand(cell_num,cell_num)/ab;
weight_outputgate_c_other=rand(cell_num,cell_num)/ab;

weight_preh_h=rand(cell_num,output_num);
weight_preh_h_other=rand(cell_num,output_num);


%weight_preh_h_other=rand(cell_num,output_num);


cost_gate=1e-30;
%cost_gate=1e-8;
%h_state=test_data;
h_state=rand(output_num, data_num);
cell_state=rand(cell_num,data_num);
yita= 0.01;
%for yita =[1000 100 10 1 ]
tau=1e-9;
%for yita=[1 10]




[stats_proposed,Accuracy_proposed, Precision_proposed, Error_iter_proposed]=LSTM_proposed(data_length,data_num, train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
bias_output_gate,ab, weight_input_x, weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x, weight_inputgate_x_other, weight_inputgate_c,weight_inputgate_c_other, weight_forgetgate_x,weight_forgetgate_x_other,...
 weight_forgetgate_c,  weight_forgetgate_c_other, weight_outputgate_x, weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other, weight_preh_h, weight_preh_h_other,  cost_gate,h_state, cell_state, yita, tau); 


[stats_proposed_signed,Accuracy_proposed_signed, Precision_proposed_signed, Error_iter_proposed_signed]=LSTM_proposed_signed(data_length,data_num, train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
bias_output_gate,ab, weight_input_x, weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x, weight_inputgate_x_other, weight_inputgate_c,weight_inputgate_c_other, weight_forgetgate_x,weight_forgetgate_x_other,...
weight_forgetgate_c,  weight_forgetgate_c_other, weight_outputgate_x, weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other, weight_preh_h, weight_preh_h_other,  cost_gate,h_state, cell_state, yita, tau); 

%[Accuracy_AMSgrad,Precision_AMSgrad, Error_iter_AMSgrad]=LSTM_AMSgrad(data_length,data_num, train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x, weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x, weight_inputgate_x_other, weight_inputgate_c,weight_inputgate_c_other, weight_forgetgate_x,weight_forgetgate_x_other,...
%weight_forgetgate_c,  weight_forgetgate_c_other, weight_outputgate_x, weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other, weight_preh_h, weight_preh_h_other,  cost_gate,h_state, cell_state, yita, tau); 



%[Accuracy,Precision, Error_iter_adam]=LSTM_ADAM(data_length,data_num, train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x, weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x, weight_inputgate_x_other, weight_inputgate_c,weight_inputgate_c_other, weight_forgetgate_x,weight_forgetgate_x_other,...
% weight_forgetgate_c,  weight_forgetgate_c_other, weight_outputgate_x, weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other, weight_preh_h, weight_preh_h_other,  cost_gate,h_state, cell_state, yita, tau); 



plot((Error_iter_proposed), 'b')
hold on
plot(Error_iter_proposed_signed, 'm')
%hold on
%plot(Error_iter_AMSgrad, 'r')
%hold on
%plot(Error_iter_adam, 'g')
%[stats_padam, Error_iter_padam]=LSTM_padam(data_length,data_num, train_data, test_data, input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c, weight_forgetgate_x,...
% weight_forgetgate_c, weight_outputgate_x, weight_outputgate_c, weight_preh_h, cost_gate,h_state, cell_state, yita); 

%[stats_proposed, Error_iter_proposed]=LSTM_proposed(data_length,data_num, train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x, weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x, weight_inputgate_x_other, weight_inputgate_c,weight_inputgate_c_other, weight_forgetgate_x,weight_forgetgate_x_other,...
 %weight_forgetgate_c,  weight_forgetgate_c_other, weight_outputgate_x, weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other, weight_preh_h, weight_preh_h_other,  cost_gate,h_state, cell_state, yita, tau); 

%[Error_iter]=test_proposed(data_length, data_num,train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c, weight_forgetgate_x,...
 %weight_forgetgate_c, weight_outputgate_x, weight_outputgate_c,weight_preh_h, cost_gate,h_state, cell_state, yita);


%end


 






%[stats_SGD, Error_iter_SGD]=LSTM_SGD(data_length, data_num,train_data, test_data, input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%,ab, weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c, weight_forgetgate_x,...
% weight_forgetgate_c, weight_outputgate_x, weight_outputgate_c,weight_preh_h, cost_gate,h_state, cell_state, yita);




%figure 
%plot((Accuracy_proposed), 'b')
%hold on
%plot(Accuracy_AMSgrad, 'm')
%hold on
%plot((Accuracy_proposed_signed), 'r')
%hold on
%plot((Accuracy_ADAM));
 
%xlabel('Epochs');
%ylabel('Accuracy');
%title('70 ms delay threshold');
%legend('Proposed','AMSgrad', 'PADAM')
%figure 
%plot((Precision_proposed), 'b')
%hold on
%plot(Precision_AMSgrad, 'm')
%hold on
%plot((Precision_proposed_signed), 'r')
%hold on
%plot((Precision_ADAM));

%xlabel('Epochs');
%ylabel('Precision');
%title('70 ms delay threshold');
%legend('Proposed','AMSgrad', 'PADAM')




%[stats_AMSgrad,Accuracy_AMSgrad,Precision_AMSgrad, Error_iter_AMSgrad]=LSTM_AMSgrad(data_length, data_num,train_data, test_data, test_hstate,test_final, input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c, weight_forgetgate_x,...
%weight_forgetgate_c, weight_outputgate_x, weight_outputgate_c,weight_preh_h, cost_gate,h_state, cell_state, 0.00001); 