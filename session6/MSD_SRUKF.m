%% MSD SR-UKF: van der Merwe and Wan, Algorithm 3.1, Eqs. (16)-(29).
% All stored factors are lower triangular: P = S*S.' and P_yy = S_y*S_y.'.
% qr(C.',0) returns an upper factor; transpose it for the paper's column factor.
% cholupdate accepts an upper factor; transpose in and transpose its result out.
% Eq. (17) uses the available previous posterior S_{k-1}; H uses X_{k|k-1}.
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
S = chol(P0, 'lower'); % Algorithm 3.1, Eq. (16): initial column factor S_0.
S_history = zeros(L, L, N); % Demo diagnostics: posterior factor history.
S_history(:, :, 1) = S; % Demo diagnostics: store S_0.
for k = 1:N-1 % Algorithm 3.1: paper time index k = 1,...,N-1.
    j = k + 1; % Storage convention: paper x_k is stored in MATLAB column k+1.
    X = repmat(x_hat(:, j-1), 1, 2*L+1) + eta*[zeros(L, 1), S, -S]; % Eq. (17): X_{k-1} from S_{k-1}.
    X_minus = F(X, u(j-1), Ad, Bd); % Eq. (18): propagate to X_{k|k-1}.
    x_hat_minus(:, j) = X_minus*Wm.'; % Eq. (19): predicted state mean x-hat_k^-.
    dX = X_minus - repmat(x_hat_minus(:, j), 1, 2*L+1); % Eqs. (20)-(21): propagated state deviations.
    Cx = [sqrt(Wc(2))*dX(:, 2:end), sqrt_Rv]; % Eq. (20): noncentral weighted columns and sqrt(R^v).
    S_minus = qr_column_factor(Cx); % Eq. (20): lower factor whose product is Cx*Cx.'.
    S_minus = signed_cholupdate(S_minus, dX(:, 1), Wc(1)); % Eq. (21): central update/downdate, scaled by W_0^(c).
    Y_minus = H*X_minus; % Eq. (22): observe the time-updated sigma states at measurement time k.
    y_hat_minus(j) = Y_minus*Wm.'; % Eq. (23): predicted observation mean y-hat_k^-.
    dY = Y_minus - y_hat_minus(j); % Eqs. (24)-(25): predicted observation deviations.
    Cy = [sqrt(Wc(2))*dY(:, 2:end), sqrt_Rn]; % Eq. (24): noncentral weighted columns and sqrt(R^n).
    S_y = qr_column_factor(Cy); % Eq. (24): lower observation-error factor.
    S_y = signed_cholupdate(S_y, dY(:, 1), Wc(1)); % Eq. (25): signed central observation contribution.
    P_xy = (dX.*repmat(Wc, L, 1))*dY.'; % Eq. (26): state/observation cross covariance.
    K = (P_xy/S_y.')/S_y; % Eq. (27): two solves give K_k*P_yy = P_xy.
    innovation(j) = y(j) - y_hat_minus(j); % Eq. (28): measurement innovation.
    x_hat(:, j) = x_hat_minus(:, j) + K*innovation(j); % Eq. (28): posterior state mean x-hat_k.
    U = K*S_y; % Eq. (28): posterior downdate columns U, satisfying U*U.' = K*P_yy*K.'.
    S = S_minus; % Eq. (29): start posterior downdates from the predicted factor.
    for ell = 1:size(U, 2) % Eq. (29): one rank-one downdate per measurement component.
        S = signed_cholupdate(S, U(:, ell), -1); % Eq. (29): internally update S.' with '-', then transpose back.
    end % Eq. (29): finish the sequential measurement downdates.
    S_history(:, :, j) = S; % Demo diagnostics: store the posterior S_k.
    P_history(:, :, j) = S*S.'; % Demo diagnostics only: reconstruct P_k; never feed it into the SR recursion.
    K_history(:, :, j) = K; % Demo diagnostics: store K_k.
end % Algorithm 3.1: complete the factor-based predict/update loop.
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
function S = qr_column_factor(C) % Paper Eqs. (20)/(24): column square root obtained by QR.
    [~, R_upper] = qr(C.', 0); % Section 3: C.' = Q*R_upper, with R_upper.'*R_upper = C*C.'.
    signs = sign(diag(R_upper)); % Factor convention: select positive diagonal entries.
    signs(signs == 0) = 1; % Factor convention: do not erase a row when the diagonal is zero.
    R_upper = diag(signs)*R_upper; % Factor convention: row sign changes preserve R_upper.'*R_upper.
    S = R_upper.'; % Paper P = S*S.': return the lower column factor.
end % QR helper: finish the covariance factor calculation.
function S = signed_cholupdate(S, column, weight) % Paper Eqs. (21)/(25)/(29): S*S.' <- S*S.' + weight*column*column.'.
    if weight == 0 % Signed update: a zero weight adds no covariance contribution.
        return; % Signed update: preserve the factor for a zero contribution.
    end % Signed update: finish the zero-weight case.
    if weight > 0 % Signed update: positive central weights add covariance.
        operation = '+'; % MATLAB cholupdate: choose a rank-one update.
    else % Signed update: negative weights subtract covariance.
        operation = '-'; % MATLAB cholupdate: choose a rank-one downdate.
    end % Signed update: finish selecting the sign.
    scaled_column = sqrt(abs(weight))*column; % Paper weighted cholupdate: scale by sqrt(|weight|).
    R_upper = cholupdate(S.', scaled_column, operation); % Eqs. (21)/(25)/(29): one output for either sign; MATLAB raises an error for an invalid downdate.
    S = R_upper.'; % Paper convention: store the updated lower factor S.
end % Signed-update helper: finish the Cholesky update/downdate.
