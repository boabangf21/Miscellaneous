
clear all;
clc;
tic  

X=load('datasetnetworkpacket2.txt');
n=size(X,1);

data_length=size(X(1:2000,1:7) ,1);
 

[Total, Delay_per_sample_1]=gen_pack_delay_seq_features(data_length, 7, (20)*1000000);
Delay_per_sample=Delay_per_sample_1(:,1:300);
train_data_1=[X(1:1000,1:7)];
train_data=train_data_1; 
test=Delay_per_sample';
test_data=[Total(1:300,:)];
%[train_data,test_data]=LSTM_data_process()



test_hstate=[Total(401:700,:)];
test_final=[X(1001:2000,:) ];


data_num=size(train_data,2);
data_length=size(train_data,1);
input_num=1000;  
cell_num=1;  
output_num=300;  



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


%cost_gate=1e-16;
cost_gate=1e-8;
%h_state=test_data;
h_state=rand(output_num, data_num);
cell_state=rand(cell_num,data_num);
for yita =[1000 100 10 1 ]
tau=1e-12;
%for yita=[1 10]


[stats_proposed,Accuracy(:, yita), Precision(:, yita), Error_iter_proposed]=LSTM_proposed(data_length,data_num, train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
bias_output_gate,ab, weight_input_x, weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x, weight_inputgate_x_other, weight_inputgate_c,weight_inputgate_c_other, weight_forgetgate_x,weight_forgetgate_x_other,...
 weight_forgetgate_c,  weight_forgetgate_c_other, weight_outputgate_x, weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other, weight_preh_h, weight_preh_h_other,  cost_gate,h_state, cell_state, 0.0001*yita, tau); 

end

%[stats_padam, Error_iter_padam]=LSTM_padam(data_length,data_num, train_data, test_data, input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c, weight_forgetgate_x,...
 %weight_forgetgate_c, weight_outputgate_x, weight_outputgate_c, weight_preh_h, cost_gate,h_state, cell_state, yita); 

%[stats_proposed_signed, Error_iter_proposed_signed]=LSTM_proposed_signed(data_length,data_num, train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x, weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x, weight_inputgate_x_other, weight_inputgate_c,weight_inputgate_c_other, weight_forgetgate_x,weight_forgetgate_x_other,...
%weight_forgetgate_c,  weight_forgetgate_c_other, weight_outputgate_x, weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other, weight_preh_h, weight_preh_h_other,  cost_gate,h_state, cell_state, yita, tau); 








%[stats_proposed, Error_iter_proposed]=LSTM_proposed(data_length,data_num, train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x, weight_input_x_other, weight_input_h, weight_input_h_other, weight_inputgate_x, weight_inputgate_x_other, weight_inputgate_c,weight_inputgate_c_other, weight_forgetgate_x,weight_forgetgate_x_other,...
 %weight_forgetgate_c,  weight_forgetgate_c_other, weight_outputgate_x, weight_outputgate_x_other, weight_outputgate_c, weight_outputgate_c_other, weight_preh_h, weight_preh_h_other,  cost_gate,h_state, cell_state, yita, tau); 

%[Error_iter]=test_proposed(data_length, data_num,train_data, test_data, test_hstate,test_final,  input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c, weight_forgetgate_x,...
 %weight_forgetgate_c, weight_outputgate_x, weight_outputgate_c,weight_preh_h, cost_gate,h_state, cell_state, yita);

%[Error_iter_adam]=LSTM_ADAM(data_length,data_num, train_data, test_data, input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c, weight_forgetgate_x,...
% weight_forgetgate_c, weight_outputgate_x, weight_outputgate_c, weight_preh_h, cost_gate,h_state, cell_state, yita); 

%end


 
%[stats_AMSgrad, Error_iter_AMSgrad]=LSTM_AMSgrad(data_length, data_num,train_data, test_data, input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%bias_output_gate,ab, weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c, weight_forgetgate_x,...
% weight_forgetgate_c, weight_outputgate_x, weight_outputgate_c,weight_preh_h, cost_gate,h_state, cell_state, yita); 







%[stats_SGD, Error_iter_SGD]=LSTM_SGD(data_length, data_num,train_data, test_data, input_num, cell_num,output_num,bias_input_gate,bias_forget_gate,.....
%,ab, weight_input_x,weight_input_h,weight_inputgate_x,weight_inputgate_c, weight_forgetgate_x,...
% weight_forgetgate_c, weight_outputgate_x, weight_outputgate_c,weight_preh_h, cost_gate,h_state, cell_state, yita);









%plot((1-Error_iter_proposed), 'b')
%hold on
%plot(Error_iter_AMSgrad, 'b')
%hold on
%plot((1-Error_iter_proposed_signed), 'm')

%y2=((1-Error_iter_proposed(30:40)));
 %x2=30:40;
%y1=(1-Error_iter_proposed_signed(30:40));
%xlabel('Epochs');
%ylabel('Accuracy');
%title('5 ms delay threshold');
%legend('Proposed', 'PADAM')
 %x1=30:40;
%grid on;
%ax1 = gca; % Store handle to axes 1.
% Create smaller axes in top right, and plot on it
% Store handle to axes 2 in ax2.
%ax2 = axes('Position',[.7 .7 .2 .2])
%box on;
%plot(x2, y2, 'b', 'LineWidth', 2)
%hold on
%plot(x1, y1, 'm', 'LineWidth', 2)
%grid on;








%hold on
figure
plot((Accuracy(:,1)))
hold on
plot((Accuracy(:,10)), 'c') 
hold on
plot ((Accuracy(:,100)),'g')

hold on
plot ((Accuracy(:,1000)),'r')
xlabel('Epochs');
ylabel('Accuracy');
title('Learning rates');

figure 
figure
plot((Precision(:,1)))
hold on
plot((Precision(:,10)), 'c') 
hold on
plot ((Precision(:,100)),'g')

hold on
plot ((Precision(:,1000)),'r')
xlabel('Epochs');
ylabel('Precision');
title('Learning rates');


  

%plot(mean(abs((final_error))));
%test_final= load('test_1.txt');
%test_final= test_final';
%test_output=test_data(:,7);  
%test_data_test=train_data(1:100,:);
%h_state_test=test_data(1:100,:);




