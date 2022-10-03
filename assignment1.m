clear;
a=readmatrix('dataset.txt'); %reading from dataset
A=a(:,2:size(a,2));
%randomizing patterns for testing and training

E=randperm(size(A,1));
train_set=A(E(1:54),:); 
test_set=A(E(55:end),:); 

input=a(:,2:5);
output=a(:,6:8);
it=10000;
m=size(input,1); %no. of patterns
n=size(input,2); %no. of input neurons
l=5; %no. of hidden neurons
k=size(output,2); %no of output neurons
eta=0.4; %learning rate
%{delw_pre=zeros(l+1,k);
delv_pre=zeros(n+1,l);
alp=0.2;
%}
in_tr_set=train_set(:,1:4); %input training data(without normalization)
out_tr_set=train_set(:,5:7); %output training data(without normalization)

in_tes_set=test_set(:,1:4); %input testing data(without normalization)
out_tes_set=test_set(:,5:7); %output testing data(without normalization)


input_norm_tr=zeros(size(train_set,1),n);
output_norm_tr=zeros(size(train_set,1),k);

input_norm_tes=zeros(size(test_set,1),n);
output_norm_tes=zeros(size(test_set,1),k);


%Normalization
for i=1:n
   input_norm_tr(:,i)=normalize(in_tr_set(:,i)); %input training data(with normalization)
   input_norm_tes(:,i)=normalize(in_tes_set(:,i));%input testing data(with normalization)
end
for i=1:k
   output_norm_tr(:,i)=0.1+0.8*normalize(out_tr_set(:,i)); %output training data
   output_norm_tes(:,i)=0.1+0.8*normalize(out_tes_set(:,i)); %output testing data
end

%training
b(1:size(input_norm_tr,1),1)=1; %bias
in_tr=[b input_norm_tr(:,1:4)]; %input training patterns with 1st coloumn as bias
le=size(in_tr,1); %number of patterns in training set

%out_tr=output_norm_tr(:,5:7); %target outputs for training data
v=rand(size(in_tr,2),l); %weight function between input layer and hidden layer
w=rand((l+1),size(output_norm_tr,2)); %weight function between hidden layer and output layer

%iteration
it_ar=zeros(it,1);
err=zeros(it,1);

for it_no=1:it
    
    it_ar(it_no,1)=it_no;
    in_hl=in_tr*v; %inputs to hidden layer
    out_hl=[b transfer(in_hl)]; %outputs of hidden layer
    in_ol=out_hl*w; %inputs to output layer
    out_ol=transfer(in_ol); %outputs from output layer
    MSE=0;
    for i=1:le
      for y=1:k
        MSE=MSE+0.5.*(out_ol(i,y)-output_norm_tr(i,y)).^2;

      end
    end
    err(it_no)=MSE./(k*le);%error calculation
 
 %training of weights
 
    delW = zeros(l+1,k) ;
    for x=1:l+1
      for y=1:k
        for z=1:le
          delW(x,y)=delW(x,y)+(output_norm_tr(z,y)-out_ol(z,y))*(1-out_ol(z,y)^2)*out_hl(z,x);
        end
      end
    end 
    
    delV = zeros(n+1,l) ;

    for x = 1:n+1
      for y=1:l
        for z=1:le
           for z1 =1:k
            delV(x,y) = delV(x,y)-(output_norm_tr(z,z1)-out_ol(z,z1))*out_ol(z,z1)*(1-out_ol(z,z1)).*w(y,z1).*out_hl(z,y).*(1-out_hl(z,y)).*in_tr(z,x) ;
           end
        end
      end
    end
    w=w+(eta/le).*delW;
    v=v+(eta/(le*k)).*delV;
   

end


% testing

 c(1:size(test_set,1),1)=1;
 in_ts=[c input_norm_tes];
 out_ts=output_norm_tes;
 in_hlt=in_ts*v;
 out_hlt=[c transfer(in_hlt)];
 in_olt=out_hlt*w;
 out_olt=transfer(in_olt);
 MSE_output=0;
 for i=1:size(test_set,1)
    for j=1:k
        MSE_output=MSE_output+(0.5/(k*size(test_set,1)))*(out_olt(i,j)-out_ts(i,j))^2;
    end
 end
 %denormalization of output
 
 for i=1:k
   output_denorm_tes(:,i)=denormalize(out_olt(:,i),out_tes_set(:,i));
 end
     output1=output_denorm_tes(:,1);
     output2=output_denorm_tes(:,2);
     output3=output_denorm_tes(:,3);
     Toutput1=out_tes_set(:,1);
     Toutput2=out_tes_set(:,2);
     Toutput3=out_tes_set(:,3);
     
     %printing output
     
figure(1)
plot(it_ar,err);
xlabel('No. of Iterations');
ylabel('Mean Square Error');
title('ERROR vs ITERATION PLOT');

figure(2)
Q=[output1';Toutput1'];
X=[1:(m-le)];
bar(X,Q);
legend('Stipulated','Experimented');
xlabel('Sample no.');
ylabel('Chip thickness(mm)');
title('CHIP THICKNESS PREDICTION');

figure(3)
Q=[output2';Toutput2'];
X=[1:(m-le)];
bar(X,Q);
legend('Stipulated','Experimented');
xlabel('Sample no.');
ylabel('Cutting force(N)');
title('CUTTING FORCE PREDICTION');


figure(4)
Q=[output3';Toutput3'];
X=[1:(m-le)];
bar(X,Q);
legend('Stipulated','Experimented');
xlabel('Sample no.');
ylabel('Feed force(N)');
title('FEED FORCE PREDICTION');





      
