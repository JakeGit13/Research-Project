function peek_mat(fpath)
% PEEK_MAT  Minimal reader: load a .mat and print what's inside.
% Usage:
%   peek_mat('results/H1/s005_a08/H1_MR+VID.mat')

    if ~exist(fpath,'file')
        error('File not found: %s', fpath);
    end

    S = load(fpath);  % load everything in the file

    % List top-level variables
    fprintf('--- %s ---\n', fpath);
    fprintf('Variables in file: %s\n', strjoin(fieldnames(S), ', '));

    % Print meta if present
    if isfield(S,'meta')
        disp('meta:');
        disp(S.meta);  % prints the struct as-is
    end

    % Print a couple of common H1 fields if present (totally optional)
    R = [];
    if isfield(S,'results'), R = S.results; end
    if isfield(S,'r1'),      R = S.r1;      end
    if ~isempty(R)
        if isfield(R,'h1_VAF_real')
            fprintf('h1_VAF_real = %.4f\n', R.h1_VAF_real);
        end
        if isfield(R,'h1_vecR_real')
            fprintf('h1_vecR_real = %.4f\n', R.h1_vecR_real);
        end
    end
end


peek_mat(fullfile(pwd,'results','H1','s005_a08','H1_MR+VID.mat'));