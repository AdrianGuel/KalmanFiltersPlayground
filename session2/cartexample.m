%% Guel Cortez 2024
% Simple one-dimensional mass-spring-damper model
%
% State:
%   x(1) = position [m]
%   x(2) = velocity [m/s]
%
% Input:
%   u = applied force [N]
%
% Measurement:
%   y = measured position [m]

%% Continuous-time model

A = [0,       1;
    -k_m/m,  -b/m];

B = [0;
     1/m];

C = [1, 0];
D = 0;

%% Exact discretisation of the deterministic model

sys_c = ss(A, B, C, D);
sys_d = c2d(sys_c, Ts, 'zoh');

Ad = sys_d.A;
Bd = sys_d.B;

%% Noise models

% Process noise:
% Unknown force acting on the cart.
%
% q_force is the continuous-time force-noise spectral density.
% Units: N^2 s, equivalently (N/sqrt(Hz))^2 under the chosen convention.
q_force = 1e-3;

% The unknown force enters through the same channel as the control force.
G = B;

% Continuous-time state-noise intensity
Qc = G*q_force*G';

% Convert continuous-time process-noise intensity into the discrete
% covariance Qd using the Van Loan method.
n = size(A,1);

M = [-A,              Qc;
      zeros(n),       A']*Ts;

E = expm(M);

Ad_noise = E(n+1:2*n, n+1:2*n)';
Qd = Ad_noise*E(1:n, n+1:2*n);

% Remove very small numerical asymmetries
Qd = (Qd + Qd')/2;

% Sensor noise:
% Standard deviation of the position sensor [m]
sigma_position = 0.1;

% Measurement-noise covariance
R = sigma_position^2;

%% Simulation

t = 0:Ts:L;
N = length(t);

x = zeros(2,N);
y = zeros(1,N);

u = 0;

% Initial true state
x(:,1) = [6;
          3];

% Initial sensor measurement
y(:,1) = C*x(:,1) + sqrt(R)*randn;

% Matrix used to generate process noise with covariance Qd
Lq = chol(Qd + 1e-12*eye(n), 'lower');

for k = 2:N

    % Process-noise sample: w_k ~ N(0,Qd)
    process_noise = Lq*randn(n,1);

    % True system
    x(:,k) = Ad*x(:,k-1) + Bd*u + process_noise;

    % Measurement-noise sample: v_k ~ N(0,R)
    measurement_noise = sqrt(R)*randn;

    % Sensor measurement
    y(:,k) = C*x(:,k) + measurement_noise;
end
