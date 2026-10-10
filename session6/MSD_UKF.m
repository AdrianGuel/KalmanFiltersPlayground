%% MSD UKF: van der Merwe and Wan, Algorithm 2.1, Eqs. (5)-(14).
% Column factors satisfy P = S*S.'; MATLAB column j stores paper time k = j-1.
% The benchmark uses acceleration-only process noise, so H*R^v = 0.
% This keeps the paper's single propagated sigma set consistent with a linear KF.
clearvars; % Demo setup: start with a fresh workspace.
close all; % Demo setup: close figures from previous runs.
Ts = 1e-2; % MSD discretization: sampling period, in seconds.
t = 0:Ts:10; % Demo setup: common time vector for both algorithms.
N = numel(t); % Demo setup: total number of stored samples.
rng(0, 'twister'); % Demo setup: repeatable data, shared by both MATLAB scripts.
m = 1; % MSD model: mass, in kg.
b = 0.5; % MSD model: viscous damping, in N s/m.
kr = 1.9; % MSD model: spring stiffness, in N/m; reserve k for the time index.
u = 0.1*sin(t); % Paper Eq. (1): known exogenous force u_k, in N.
w = 1e-4; % Demo model: standard deviation of sampled acceleration noise, in m/s^2.
v = 1e-3; % Demo model: standard deviation of position measurement noise, in m.
L = 2; % Paper Section 2: state dimension for x_k = [position; velocity].
A = [0, 1; -kr/m, -b/m]; % MSD model: continuous-time state matrix.
B = [0; 1/m]; % MSD model: continuous-time input matrix.
Ad = eye(L) + Ts*A; % Paper Eq. (1): forward-Euler state transition for F.
Bd = Ts*B; % Paper Eq. (1): forward-Euler input map for u_{k-1}.
H = [1, 0]; % Paper Eq. (2): position-only observation map H[x_k].
sqrt_Rv = diag([0, Ts*w]); % Paper R^v: column square root of acceleration-induced state noise.
Rv = sqrt_Rv*sqrt_Rv.'; % Paper R^v: covariance of the discrete process noise v_k.
Rn = v^2; % Paper R^n: covariance of the measurement noise n_k.
sqrt_Rn = sqrt(Rn); % Paper sqrt(R^n): scalar measurement-noise square root.
x = zeros(L, N); % Demo data: true state history; column j corresponds to paper k = j-1.
y = zeros(1, N); % Paper Eq. (2): measured position history y_k.
x(:, 1) = [1; 0.1]; % Demo data: true initial state x_0.
y(1) = H*x(:, 1) + sqrt_Rn*randn; % Paper Eq. (2): initial measurement, stored for plotting.
for j = 2:N % Demo data: simulate paper transitions k-1 -> k.
    x(:, j) = F(x(:, j-1), u(j-1), Ad, Bd) + sqrt_Rv*randn(L, 1); % Paper Eq. (1): F[x_{k-1},u_{k-1}] + v_{k-1}.
    y(j) = H*x(:, j) + sqrt_Rn*randn; % Paper Eq. (2): H[x_k] + n_k.
end % Demo data: complete the simulation loop.
alpha = 0.99; % Paper Section 2: sigma-point spread parameter.
beta = 2; % Paper Section 2: Gaussian prior choice for the covariance weight.
lambda = L*(alpha^2 - 1); % Paper Section 2: lambda = L(alpha^2 - 1), with no extra kappa parameter.
eta = sqrt(L + lambda); % Paper Section 2: sigma-point scaling eta.
Wm = [lambda/(L + lambda), repmat(1/(2*(L + lambda)), 1, 2*L)]; % Paper W_i^(m), ordered i = 0,...,2L.
Wc = Wm; % Paper W_i^(c): noncentral covariance weights equal mean weights.
Wc(1) = Wm(1) + 1 - alpha^2 + beta; % Paper W_0^(c): central covariance-weight correction.
x_hat = zeros(L, N); % Paper x-hat_k: posterior state estimates.
x_hat(:, 1) = [0.1; 0.2]; % Paper initialization: x-hat_0; y_0 is not assimilated again.
P0 = diag([0.5, 0.18]); % Paper initialization: covariance P_0 of the initial estimation error.
x_hat_minus = nan(L, N); % Paper x-hat_k^-: predicted state estimates.
y_hat_minus = nan(1, N); % Paper y-hat_k^-: predicted observations.
P_history = zeros(L, L, N); % Demo diagnostics: posterior covariance history.
P_history(:, :, 1) = P0; % Demo diagnostics: store the initial covariance.
K_history = nan(L, 1, N); % Paper K_k: Kalman-gain history.
innovation = nan(1, N); % Paper Eqs. (13)/(28): innovations y_k - y-hat_k^-.
P = P0; % Algorithm 2.1, Eq. (5): initialize the full posterior covariance.
for k = 1:N-1 % Algorithm 2.1: paper time index k = 1,...,N-1.
    j = k + 1; % Storage convention: paper x_k is stored in MATLAB column k+1.
    sqrt_P = chol(P, 'lower'); % Algorithm 2.1, Eq. (6): column square root of P_{k-1}.
    X = repmat(x_hat(:, j-1), 1, 2*L+1) + eta*[zeros(L, 1), sqrt_P, -sqrt_P]; % Eq. (6): X_{k-1}, with central column first.
    X_minus = F(X, u(j-1), Ad, Bd); % Eq. (7): X_{k|k-1} = F[X_{k-1},u_{k-1}].
    x_hat_minus(:, j) = X_minus*Wm.'; % Eq. (8): weighted predicted mean x-hat_k^-.
    dX = X_minus - repmat(x_hat_minus(:, j), 1, 2*L+1); % Eq. (9): propagated state deviations.
    P_minus = (dX.*repmat(Wc, L, 1))*dX.' + Rv; % Eq. (9): P_k^- = sum W_i^(c) dX_i dX_i.' + R^v.
    Y_minus = H*X_minus; % Eq. (9): Y_{k|k-1} = H[X_{k|k-1}].
    y_hat_minus(j) = Y_minus*Wm.'; % Eq. (10): weighted observation prediction y-hat_k^-.
    dY = Y_minus - y_hat_minus(j); % Eq. (11): predicted observation deviations.
    P_yy = (dY.*Wc)*dY.' + Rn; % Eq. (11): P_{tilde y_k tilde y_k}, including R^n.
    P_xy = (dX.*repmat(Wc, L, 1))*dY.'; % Eq. (11): P_{x_k y_k} from paired propagated deviations.
    K = P_xy/P_yy; % Eq. (12): K_k = P_{x_k y_k}/P_{tilde y_k tilde y_k}; no explicit inverse.
    innovation(j) = y(j) - y_hat_minus(j); % Eq. (13): measurement innovation y_k - y-hat_k^-.
    x_hat(:, j) = x_hat_minus(:, j) + K*innovation(j); % Eq. (13): posterior mean x-hat_k.
    P = P_minus - K*P_yy*K.'; % Eq. (14): posterior covariance P_k.
    P = 0.5*(P + P.'); % Numerical implementation of Eq. (14): remove antisymmetric roundoff only.
    P_history(:, :, j) = P; % Demo diagnostics: store P_k after the measurement update.
    K_history(:, :, j) = K; % Demo diagnostics: store K_k.
end % Algorithm 2.1: complete one predict/update cycle per measurement.
error_state = x - x_hat; % Demo diagnostics: estimation error relative to the simulated truth.
rmse = sqrt(mean(error_state(:, 2:end).^2, 2)); % Demo diagnostics: RMSE after the first measurement update.
fprintf('Position RMSE: %.6g m; velocity RMSE: %.6g m/s\n', rmse(1), rmse(2)); % Demo diagnostics: report estimation accuracy.
figure('Color', 'w'); % Demo visualization: create the state-estimation figure.
subplot(2, 1, 1); % Demo visualization: position panel.
plot(t, x(1, :), 'k-', t, y, '.', t, x_hat(1, :), 'b-'); % Demo visualization: true, measured, and estimated position.
grid on; % Demo visualization: show the position grid.
xlabel('Time [s]'); % Demo visualization: label the horizontal axis.
ylabel('Position [m]'); % Demo visualization: label the physical quantity.
legend('True position', 'Measurement y_k', 'Estimated position', 'Location', 'best'); % Demo visualization: identify curves.
subplot(2, 1, 2); % Demo visualization: velocity panel.
plot(t, x(2, :), 'k-', t, x_hat(2, :), 'b-'); % Demo visualization: true and estimated unmeasured velocity.
grid on; % Demo visualization: show the velocity grid.
xlabel('Time [s]'); % Demo visualization: label the horizontal axis.
ylabel('Velocity [m/s]'); % Demo visualization: label the physical quantity.
legend('True velocity', 'Estimated velocity', 'Location', 'best'); % Demo visualization: identify curves.
figure('Color', 'w'); % Demo visualization: create the error and uncertainty figure.
for ell = 1:L % Demo diagnostics: plot each state component's estimation error.
    subplot(L, 1, ell); % Demo visualization: select this state component's panel.
    sigma = sqrt(reshape(P_history(ell, ell, :), 1, N)); % Paper covariance: marginal posterior standard deviation.
    plot(t, error_state(ell, :), 'b-', t, 3*sigma, 'k--', t, -3*sigma, 'k--'); % Demo visualization: error and +/-3 sigma envelopes.
    grid on; % Demo visualization: show the error grid.
    xlabel('Time [s]'); % Demo visualization: label the horizontal axis.
    ylabel(sprintf('State %d error', ell)); % Demo visualization: identify the state error.
    legend('True - estimated', '+3 sigma', '-3 sigma', 'Location', 'best'); % Demo visualization: identify the diagnostic curves.
end % Demo diagnostics: finish plotting both state components.
function X_next = F(X, u_previous, Ad, Bd) % Paper Eqs. (7)/(18): deterministic transition, applied to every column.
    X_next = Ad*X + repmat(Bd*u_previous, 1, size(X, 2)); % MSD Euler map F[X,u]; random noise is represented separately by R^v.
end % Model helper: finish F.
