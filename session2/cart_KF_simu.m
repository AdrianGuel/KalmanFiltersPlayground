clearvars;

[y_std, ] = cartexample([6,0]');

m = 1;
b = 0.3;
A= [[0,1];[0,-b/m]];
B= [0,1/m]';
C= [1,0];
D = 0;
Ts = 1e-3;

%Euler method
Ad = eye(length(A))+Ts*A;
Bd = Ts*B;

L=10;
t=0:Ts:L;
Qv = 1e-6*eye(2);
Qxi = std(y_std)^2;
P = 10*eye(2);

x_pred=zeros([2,length(t)]);
y_pred=zeros([1,length(t)]);
x_est=zeros([2,length(t)]);
u = 0;
[y,x] = cartexample([6,5]');
alpha = 1e-2;
x_est(:,1) = [2,0]';
for k=2:length(t)
    x_pred(:,k)=Ad*x_est(:,k-1)+Bd*u;
    y_pred(k)=C*x_pred(:,k);

    P_pred=Ad*P*Ad' + Qv;
    P_y=C*P_pred*C' + Qxi;
    P_xy=P_pred*C';

    L=P_xy/P_y;
    x_est(:,k)=x_pred(:,k)+L*(y(k)-y_pred(k));
    P=P_pred-L*P_y*L';

    Qv=(1-alpha)*Qv+alpha*L*(y(k)-y_pred(k))*(y(k)-y_pred(k))'*L';
end

figure
subplot(2,1,1)
plot(t,y)
hold on
plot(t,x_est(1,:),'r')

subplot(2,1,2)
plot(t,x_est(2,:))
hold on
plot(t,x(2,:))
