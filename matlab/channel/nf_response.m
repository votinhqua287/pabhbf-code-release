function B = nf_response(r, theta, Nt, lam)
%NF_RESPONSE Non-uniform spherical-wave response of a centered ULA on the y-axis.
%   B = NF_RESPONSE(r, theta, Nt, lam) returns an S-by-Nt matrix whose rows are the responses toward the
%   points (r(i), theta(i)). [b]_n = (r / r_n) exp(-j 2 pi (r_n - r) / lam) / sqrt(Nt),
%   r_n = sqrt(r^2 + delta_n^2 d^2 - 2 r delta_n d sin(theta)), delta_n = n - (Nt+1)/2, d = lam/2.
d = lam / 2;
delta = (1:Nt) - (Nt + 1) / 2;
r = r(:);
theta = theta(:);
dist = sqrt(r.^2 + (delta * d).^2 - 2 * r .* (delta * d) .* sin(theta));
B = (r ./ dist) .* exp(-1j * 2 * pi / lam * (dist - r)) / sqrt(Nt);
end
