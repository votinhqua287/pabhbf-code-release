function Fbar = rzf_directions(Hbar, reg)
%RZF_DIRECTIONS Regularized zero-forcing directions Fbar = Hbar^H (Hbar Hbar^H + reg I)^(-1).
%   Hbar is S-by-S (effective channel Hhat_S^H F_RF); reg = 0 gives zero forcing.
S = size(Hbar, 1);
Fbar = Hbar' / (Hbar * Hbar' + reg * eye(S));
end
