function inspect_h1_result(resultsRoot, sentenceID, actorID, whichRun)
%INSPECT_H1_RESULT  Quick-print H1 result summary for a given (sentence, actor).
% Usage:
%   inspect_h1_result('results/H1', 5, 8, 'MR+VID');
%   inspect_h1_result('results/H1', 5, 8, 'MRonly');
%   inspect_h1_result('results/H1', 5, 8, 'VIDonly');

    if nargin < 4, whichRun = 'MR+VID'; end

    % Map shorthand -> filename
    switch upper(whichRun)
        case 'MR+VID', fname = 'H1_MR+VID.mat';
        case 'MRONLY', fname = 'H1_MRonly.mat';
        case 'VIDONLY', fname = 'H1_VIDonly.mat';
        otherwise, error('Unknown whichRun: %s', whichRun);
    end

    folder = fullfile(resultsRoot, sprintf('s%03d_a%02d', sentenceID, actorID));
    fpath  = fullfile(folder, fname);

    if ~exist(fpath, 'file')
        error('File not found: %s', fpath);
    end

    % Load just what we need
    S = load(fpath);  % contains either 'results' or 'r1' + 'meta'

    % Normalize variable name
    if isfield(S, 'results')
        R = S.results;
    elseif isfield(S, 'r1')
        R = S.r1;
    else
        error('No ''results'' or ''r1'' variable in MAT file.');
    end

    M = [];
    if isfield(S, 'meta'), M = S.meta; end

    % --- Print meta ---
    fprintf('--- %s ---\n', fpath);
    if ~isempty(M)
        fprintf('actorID=%s | sentenceID=%s | dataIdx=%s | observedMode=%s | nBoots=%s\n', ...
            strnum(M,'actorID'), strnum(M,'sentenceID'), strnum(M,'dataIdx'), strstr(M,'observedMode'), strnum(M,'nBoots'));
        if isfield(M, 'reconstructId'), fprintf('reconstructId=%s | ', strnum(M,'reconstructId')); end
        if isfield(M, 'shuffleTarget'), fprintf('shuffleTarget=%s | ', strnum(M,'shuffleTarget')); end
        if isfield(M, 'rngSeed'),       fprintf('rngSeed=%s | ',       strnum(M,'rngSeed'));       end
        if isfield(M, 'timestamp'),     fprintf('timestamp=%s',       M.timestamp);               end
        fprintf('\n');
    end

    % --- Pick common H1 fields (robust to names) ---
    vaf   = pick(R, {'h1_VAF_real','VAF_real','VAF'});
    ci    = pick(R, {'h1_refit_VAF_ci','refit_VAF_CI','VAF_CI'});
    pVAF  = pick(R, {'h1_refit_VAF_p','refit_VAF_p','VAF_p'});
    vecR  = pick(R, {'h1_vecR_real','vecR_real','vecR'});
    pVecR = pick(R, {'h1_vecR_p','vecR_p'});

    % --- Print summary ---
    fprintf('H1 summary:\n');
    fprintf('  VAF = %s%%', pct(vaf));
    if ~isempty(ci) && numel(ci)>=2
        fprintf(' | 95%% CI = [%s%%, %s%%]', pct(ci(1)), pct(ci(2)));
    end
    if ~isempty(pVAF),  fprintf(' | p_VAF = %.3g', pVAF); end
    fprintf('\n');
    if ~isempty(vecR)
        fprintf('  vecR = %.3f', vecR);
        if ~isempty(pVecR), fprintf(' | p_vecR = %.3g', pVecR); end
        fprintf('\n');
    end

    % If you ever want to peek at fields:
    % disp(fieldnames(R));

end

% --------- helpers ----------
function out = pick(S, names)
    out = [];
    for i = 1:numel(names)
        f = names{i};
        if isfield(S, f)
            out = S.(f);
            return
        end
    end
end

function s = pct(x)
    if isempty(x) || ~isnumeric(x), s = 'n/a'; return; end
    if max(x) <= 1, x = x*100; end
    s = sprintf('%.1f', x);
end

function s = strnum(M, f)
    if ~isfield(M,f), s = 'n/a'; return; end
    v = M.(f);
    if ischar(v) || isstring(v), s = char(v); return; end
    if isnumeric(v), s = num2str(v); return; end
    s = 'n/a';
end

function s = strstr(M, f)
    if ~isfield(M,f), s = 'n/a'; return; end
    s = char(string(M.(f)));
end
