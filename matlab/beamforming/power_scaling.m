function FBB = power_scaling(FRF, Fbar, shares)
%POWER_SCALING Scale the digital directions so that ||F_RF f_k||^2 = shares(k) (Pmax = 1).
col = vecnorm(FRF * Fbar, 2, 1);
assert(all(col > 0), 'power_scaling:degenerate', 'zero precoder column');
FBB = Fbar .* (sqrt(shares(:).') ./ col);
end
