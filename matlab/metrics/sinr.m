function gamma = sinr(H_S, FRF, FBB, sigma2)
%SINR SINR of the scheduled users treating interference as noise.
%   H_S is Nt-by-S (columns = true channels of the scheduled users), FRF Nt-by-S, FBB S-by-S.
assert(sigma2 > 0, 'sinr:noise', 'noise variance must be positive');
G = H_S' * FRF * FBB;
P = abs(G).^2;
sig = diag(P);
gamma = sig ./ (sum(P, 2) - sig + sigma2);
assert(all(isfinite(gamma)), 'sinr:nonfinite', 'non-finite SINR');
end
