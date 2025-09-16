function summary = qc_H1_results(resultsRoot)
% Minimal QC for H1 outputs (no plots, no fluff).

if nargin==0 || isempty(resultsRoot)
    resultsRoot = 'C:\Users\jaker\Research-Project\data\results';
end
h1Root = fullfile(resultsRoot,'H1');

kinds = { ...
  'H1_MR+VID.mat', 'MR+VID', 1; ...
  'H1_MR_ONLY.mat','MR',     1; ...
  'H1_VID_ONLY.mat','VID',   2  ...
}; % file, observedMode (expected), shuffleTarget (expected)

D = dir(h1Root); D = D([D.isdir]);
D = D(~ismember({D.name},{'.','..'}));

rows = [];
present = 0; missing = 0;

fprintf('\n=== H1 QC @ %s ===\n', h1Root);

for di = 1:numel(D)
    sub = D(di).name;  % e.g., s256_a08
    for ki = 1:size(kinds,1)
        f = fullfile(h1Root, sub, kinds{ki,1});
        expectedMode   = kinds{ki,2};
        expectedShufT  = kinds{ki,3};
        if ~exist(f,'file')
            missing = missing + 1;
            continue;
        end
        present = present + 1;

        % Load robustly
        clear r meta
        S = load(f);
        if isfield(S,'r1'),      r = S.r1;
        elseif isfield(S,'results'), r = S.results;
        else, r = struct(); 
        end
        if isfield(S,'meta'), meta = S.meta; else, meta = struct(); end

        % Meta (best-effort)
        obsMode   = safeGet(meta,'observedMode', expectedMode);
        shufT     = safeGet(meta,'shuffleTarget', NaN);
        reconId   = safeGet(meta,'reconstructId', NaN);
        nBoots    = safeGet(meta,'nBoots', NaN);
        actorID   = safeGet(meta,'actorID', NaN);
        sentenceID= safeGet(meta,'sentenceID', NaN);

        % H1 metrics
        VAF_real   = safeGet(r,'h1_VAF_real', NaN);
        vecR_real  = safeGet(r,'h1_vecR_real', NaN);

        refitShuffs= safeGet(r,'h1_refit_VAF_shuffs', []);
        p_VAF_refit= safeGet(r,'h1_refit_VAF_p', NaN);
        med_refit  = safeMed(refitShuffs);

        evalShuffs = safeGet(r,'h1_eval_VAF_shuffs', []);
        p_VAF_eval = safeGet(r,'h1_eval_VAF_p', NaN);
        med_eval   = safeMed(evalShuffs);

        vecR_shuffs= safeGet(r,'h1_vecR_shuff_all', []);
        p_vecR     = safeGet(r,'h1_vecR_p', NaN);

        boots_inferred = max([numel(refitShuffs), numel(evalShuffs), numel(vecR_shuffs)]);
        if isnan(nBoots), nBoots = boots_inferred; end

        flags = {};
        withinVAF = @(x) all(x(~isnan(x)) > -1 & x(~isnan(x)) < 1.000001); % allow slight FP >1
        if ~(reconId==3),                              flags{end+1}='reconId!=3'; end
        if ~strcmp(obsMode,expectedMode),              flags{end+1}='obsMode!=expected'; end
        if ~(isnan(shufT) || shufT==expectedShufT),    flags{end+1}='shuffleTarget!=expected'; end
        if ~withinVAF([VAF_real med_refit med_eval]),  flags{end+1}='VAF outside [-1,1]'; end
        if ~isfinite(VAF_real),                        flags{end+1}='VAF_real NaN'; end
        if boots_inferred==0,                          flags{end+1}='no shuffles'; end

        pass_refit = isfinite(VAF_real) && isfinite(med_refit) && (VAF_real > med_refit) && isfinite(p_VAF_refit) && (p_VAF_refit <= 0.05);
        pass_eval  = isfinite(VAF_real) && isfinite(med_eval)  && (VAF_real > med_eval)  && isfinite(p_VAF_eval)  && (p_VAF_eval  <= 0.05);

        passTxt = 'PASS';
        if ~pass_refit, passTxt='CHECK'; end
        if ~isempty(flags), passTxt=[passTxt ' [' strjoin(flags,',') ']']; end

        fprintf('%-12s | %-7s | VAF=%.3f (ref med=%.3f, p=%.3g) | vecR=%.3f (p=%.3g) | nB=%4d | %s\n',...
            sub, obsMode, VAF_real, med_refit, p_VAF_refit, vecR_real, p_vecR, nBoots, passTxt);

        rows = [rows; struct( ...
            'folder',sub, 'mode',obsMode, 'actor',actorID, 'sentence',sentenceID, ...
            'VAF',VAF_real, 'VAF_med_refit',med_refit, 'p_VAF_refit',p_VAF_refit, ...
            'VAF_med_eval',med_eval, 'p_VAF_eval',p_VAF_eval, ...
            'vecR',vecR_real, 'p_vecR',p_vecR, ...
            'nBoots',nBoots, 'pass_refit',pass_refit, 'pass_eval',pass_eval, ...
            'flags',{flags})]; %#ok<AGROW>
    end
end

nPass = sum([rows.pass_refit]);
fprintf('\nPresent files: %d | Missing expected files: %d\n', present, missing);
fprintf('PASS (refit criterion): %d / %d\n', nPass, numel(rows));
summary = rows;

end

% ---------- helpers ----------
function val = safeGet(s, fld, def)
    if isstruct(s) && isfield(s, fld)
        val = s.(fld);
    else
        val = def;
    end
end

function m = safeMed(x)
    if isempty(x), m = NaN; else, m = median(x,'omitnan'); end
end

summary = qc_H1_results('C:\Users\jaker\Research-Project\data\results');
