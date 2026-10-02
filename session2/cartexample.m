function [y,x] = cartexample(x0)
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

x = zeros(length(A),length(t));
y = zeros(1,length(t));
u = 0;

x(:,1)= x0;
v = 1e-4;
xi = 1e-1;

for k=2:length(t)
    x(:,k)= Ad*x(:,k-1)+Bd*u + v*randn([2,1]);
    y(k) = C*x(:,k) + xi*randn;
end
figure
plot(t,y)

end
