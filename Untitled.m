
    
v=load('ENDTOEND.txt');
d=v';
mean=(max(v)-min(v))/10;

[Delay_status_for_1, Delay_status_per_feature_1]=gen_pack_delay_seq_features(451, 1, 248.0269);

[Delay_status_for_2, Delay_status_per_feature_2]=gen_pack_delay_seq_features(451, 1, 248.2478);

[Delay_status_for_3, Delay_status_per_feature_3]=gen_pack_delay_seq_features(451, 1, 248.4688);


[Delay_status_for_4, Delay_status_per_feature_4]=gen_pack_delay_seq_features(451, 1, 248.6263);

[Delay_status_for_5, Delay_status_per_feature_5]=gen_pack_delay_seq_features(451, 1, 248.649);

[Delay_status_for_6, Delay_status_per_feature_6]=gen_pack_delay_seq_features(451, 1, 248.6735);



[Delay_status_for_7, Delay_status_per_feature_7]=gen_pack_delay_seq_features(451, 1, 248.6898);

[Delay_status_for_8, Delay_status_per_feature_8]=gen_pack_delay_seq_features(451, 1, 248.7482);

[Delay_status_for_9, Delay_status_per_feature_9]=gen_pack_delay_seq_features(451, 1, 248.7574);

[Delay_status_for_10, Delay_status_per_feature_10]=gen_pack_delay_seq_features(451, 1, 248.7756);

[Delay_status_for_11, Delay_status_per_feature_11]=gen_pack_delay_seq_features(451, 1, 248.7921);

[Delay_status_for_12, Delay_status_per_feature_12]=gen_pack_delay_seq_features(451, 1, 248.9108);

[Delay_status_for_13, Delay_status_per_feature_13]=gen_pack_delay_seq_features(451, 1, 249.1318);

[Delay_status_for_14, Delay_status_per_feature_14]=gen_pack_delay_seq_features(451, 1, 249.3527);

[Delay_status_for_15, Delay_status_per_feature_15]=gen_pack_delay_seq_features(451, 1, 249.5737 );

[Delay_status_for_16, Delay_status_per_feature_16]=gen_pack_delay_seq_features(451, 1, 249.7947);

[Delay_status_for_17, Delay_status_per_feature_17]=gen_pack_delay_seq_features(451, 1,  250.0157);

[Delay_status_for_18, Delay_status_per_feature_18]=gen_pack_delay_seq_features(451, 1,  250.2366);


Total= [ nnz(~Delay_status_for_1)/numel(v), nnz(~Delay_status_for_2)/numel(v), nnz(~Delay_status_for_3)/numel(v), nnz(~Delay_status_for_4)/numel(v), nnz(~Delay_status_for_5)/numel(v), nnz(~Delay_status_for_6)/numel(v), nnz(~Delay_status_for_7)/numel(v), nnz(~Delay_status_for_8)/numel(v), nnz(~ Delay_status_for_9)/numel(v), nnz(~Delay_status_for_10)/numel(v), nnz(~ Delay_status_for_11)/numel(v), nnz(~ Delay_status_for_12)/numel(v), nnz(~ Delay_status_for_13)/numel(v), nnz(~ Delay_status_for_14)/numel(v), nnz(~ Delay_status_for_15)/numel(v).....
 nnz(~Delay_status_for_16)/numel(v), nnz(~Delay_status_for_17)/numel(v), nnz(~Delay_status_for_18)/numel(v)  ];
i=[248.03 248.23  248.47 248.63 248.65  248.67  248.69 248.75  248.76  248.78 248.79 248.91 249.13 249.35 249.57 249.79 250.02 250.25  ];
plot(i, Total,'-o')
xlabel('Delay threshold [ms]')
ylabel('Delayed packets')
%plot(i, Total)

%W=load('endtoenddelay.csv')
W=load('ENDTOEND.txt')
histogram(W)
xlabel('End-to-end delay [ms]')
ylabel('Frequency')
