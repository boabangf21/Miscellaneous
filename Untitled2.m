


i=[248.0269 248.2478  248.4688 248.6263 248.649  248.6735  248.6898 248.7482  248.7574  248.7756 248.7921 ...
    248.9108 249.1318 249.3527 249.5737   ];
B=load('result_7.csv');
figure
plot(i, 100*B(:,2), 'b--*')
hold on
plot(i, 100*B(:,1), 'r-o')


legend('1-packet-ahead prediction','2-packet-ahead prediction')
xlabel('Delay threshold [ms]')
ylabel('Accuracy (%)')



C=load('9a_figure.csv');

figure

axis1= categorical( {'248.47ms', '248.76ms', '250.24ms','248.47 ms', '248.76 ms', '250.24 ms'});
axis1 = reordercats(axis1,cellstr(axis1)');
Y=[3.557720423 3.407950878 3.492495775 2.133426666 2.140779734 1.968128204];
h1(1)=bar(axis1(1), Y(1), 'r')
hold on
bar(axis1(2), Y(2), 'r')
hold on
bar(axis1(3), Y(3), 'r')
hold on
h2(1)=bar(axis1(4), Y(4), 'b')
hold on
bar(axis1(5), Y(5), 'b')
hold on
bar(axis1(6), Y(6), 'b')
%legend([h1(1), h2(1)], 'DATASET1', ' DATASET2')
legend([h1(1), h2(1)],'Proposed SGD (1-packet-ahead prediction)','ADAM (1-packet-ahead prediction)')
xlabel('Delay threshold')
ylabel('Training time [s]')

figure

Y_two_step=[3.637549877166748 3.369137763977051 3.12070894241333  1.680342674255371 1.8308053016662598 1.8956542015075684 ]

axis1= categorical([ "248.47ms", "248.76ms", "250.24ms", "248.47 ms", "248.76 ms", "250.24 ms"]);
axis1 = reordercats(axis1,cellstr(axis1)');

h2(1)=bar(axis1(4), Y_two_step(4), 'b')
hold on
bar(axis1(5), Y_two_step(5), 'b')
hold on
bar(axis1(6), Y_two_step(6), 'b')
hold on
h1(1)=bar(axis1(1), Y_two_step(1), 'r')
hold on
bar(axis1(2), Y_two_step(2), 'r')
hold on
bar(axis1(3), Y_two_step(3), 'r')
hold on

legend([h1(1), h2(1)],'Proposed SGD (2-packet-ahead prediction)','ADAM (2-packet-ahead prediction)')
%legend([h1(1), h2(1)],'{LSTM with proposed SGD for 2 step ahead packet prediction}','{LSTM with ADAM for 2 step ahead packet prediction}')
xlabel('Delay threshold')
ylabel('Training time [s]')

%figure
%W=load('endtoenddelay.csv')
%histogram(W)
%xlabel('Delay threshold [ms]')
%ylabel('Frequency')


