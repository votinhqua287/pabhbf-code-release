function crosscheck_python(in_file, out_file)
%CROSSCHECK_PYTHON Recompute channels and rates from the inputs exported by pabhbf.eval.crosscheck_matlab.
%   The Python side exports geometry, path gains, estimates, assigned codewords, schedules, power shares and SNRs;
%   this function recomputes H with the MATLAB channel model and the rates with RZF, power scaling and SINR.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'channel'), fullfile(here, '..', 'beamforming'), fullfile(here, '..', 'metrics'));
S = load(in_file);
[n, K, ~] = size(S.r_paths);
Nt = double(S.Nt);
lam = double(S.lam);
H = zeros(n, K, Nt);
for i = 1:n
    for k = 1:K
        H(i, k, :) = combine_paths(squeeze(S.alpha(i, k, :)), squeeze(S.r_paths(i, k, :)), ...
            squeeze(S.theta_paths(i, k, :)), Nt, lam, 'near_field');
    end
end
Ns = size(S.sched, 2);
rates = zeros(n, Ns);
for i = 1:n
    sched = double(S.sched(i, :)) + 1;
    Hhat_S = reshape(S.Hhat(i, sched, :), Ns, Nt).';
    H_S = reshape(H(i, sched, :), Ns, Nt).';
    FRF = S.W(double(S.idx(i, :)) + 1, :).';
    snr = double(S.snr(i));
    Fbar = rzf_directions(Hhat_S' * FRF, Ns / snr);
    FBB = power_scaling(FRF, Fbar, S.shares(i, :));
    rates(i, :) = log2(1 + sinr(H_S, FRF, FBB, 1 / snr)).';
end
save(out_file, 'H', 'rates');
fprintf('crosscheck_python: wrote %s\n', out_file);
end
