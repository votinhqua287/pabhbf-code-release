function B = ff_response(theta, Nt, lam)
%FF_RESPONSE Far-field (plane-wave) steering vectors of a centered ULA on the y-axis.
%   B = FF_RESPONSE(theta, Nt, lam) returns an S-by-Nt matrix, [b]_n = exp(+j 2 pi delta_n d sin(theta)/lam)/sqrt(Nt).
d = lam / 2;
delta = (1:Nt) - (Nt + 1) / 2;
theta = theta(:);
B = exp(1j * 2 * pi / lam * sin(theta) * (delta * d)) / sqrt(Nt);
end
