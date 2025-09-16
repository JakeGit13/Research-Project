function peek_mat(fpath)
    % Usage: peek_mat('results/H1/s005_a08/H1_MR+VID.mat')
    if ~exist(fpath,'file'); error('File not found: %s', fpath); end
    S = load(fpath);             % load everything
    fprintf('--- %s ---\n', fpath);
    disp(fieldnames(S));         % show top-level variables
    if isfield(S,'meta'); disp(S.meta); end


    S = load(fpath);
    S.r1.h1_VAF_real
    S.r1.h1_refit_VAF_ci
    S.r1.h1_refit_VAF_p
    S.r1.h1_vecR_real
    S.r1.h1_vecR_p
end


