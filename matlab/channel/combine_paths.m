function h = combine_paths(alpha, r_paths, theta_paths, Nt, lam, model)
%COMBINE_PATHS Channel of one user from its path gains and path geometry.
%   h = COMBINE_PATHS(alpha, r_paths, theta_paths, Nt, lam, model) with alpha, r_paths, theta_paths of length L+1
%   (index 1 = LoS) returns the Nt-by-1 channel h = sqrt(Nt) * sum_l alpha_l b_l, model = 'near_field' | 'far_field'.
switch model
    case 'near_field'
        B = nf_response(r_paths, theta_paths, Nt, lam);
    case 'far_field'
        B = ff_response(theta_paths, Nt, lam);
    otherwise
        error('combine_paths:model', 'unknown model %s', model);
end
h = sqrt(Nt) * (alpha(:).' * B).';
end
