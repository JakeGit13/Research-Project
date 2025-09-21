function summary = analyse_h1_results(resultsRoot)
% QC for H1 outputs with folder structure:
% results\H1\actor_##\s###\H1_*.mat


clc;

h1Root = fullfile(resultsRoot,'H1');

% Expected filenames and their metadata expectations
kinds = { ...
  'H1_MR+VID.mat', 'MR+VID', 1; ...
  'H1_MR_ONLY.mat','MR',     1; ...
  'H1_VID_ONLY.mat','VID',   2 ...
};



% Discover actor folders
Dactors = dir(h1Root); 
Dactors = Dactors([Dactors.isdir]);
Dactors = Dactors(~ismember({Dactors.name},{'.','..'}));

rows = [];
present = 0; missing = 0;

fprintf('\n=== H1 QC @ %s ===\n', h1Root);

for ai = 1:numel(Dactors)
    actorFolder = Dactors(ai).name;  % e.g., actor_08
    actorPath   = fullfile(h1Root, actorFolder);

    % Sentence subfolders
    S = dir(actorPath);
    S = S([S.isdir]);
    S = S(~ismember({S.name},{'.','..'}));

    for si = 1:numel(S)
        sentFolder = S(si).name;  % e.g., s001
        subPath    = fullfile(actorPath, sentFolder);

        for ki = 1:size(kinds,1)
            f = fullfile(subPath, kinds{ki,1});
            expectedMode   = kinds{ki,2};
            expectedShufT  = kinds{ki,3};

            if ~exist(f,'file')
                missing = missing + 1;
                continue;
            end
            present = present + 1;

            % Load results
            clear r meta
            Sload = load(f);
            if isfield(Sload,'r1'),      r = Sload.r1;
            elseif isfield(Sload,'results'), r = Sload.results;
            else, r = struct(); 
            end
            if isfield(Sload,'meta'), meta = Sload.meta; else, meta = struct(); end

            % Meta
            obsMode    = safeGet(meta,'observedMode', expectedMode);
            shufT      = safeGet(meta,'shuffleTarget', NaN);
            reconId    = safeGet(meta,'reconstructId', NaN);
            nBoots     = safeGet(meta,'nBoots', NaN);
            actorID    = safeGet(meta,'actorID', NaN);
            sentenceID = safeGet(meta,'sentenceID', NaN);
            dataIdx    = safeGet(meta,'dataIdx', NaN);

            % Metrics
            VAF_real    = safeGet(r,'h1_VAF_real', NaN);
            vecR_real   = safeGet(r,'h1_vecR_real', NaN);
            refitShuffs = safeGet(r,'h1_refit_VAF_shuffs', []);
            p_VAF_refit = safeGet(r,'h1_refit_VAF_p', NaN);
            med_refit   = safeMed(refitShuffs);
            evalShuffs  = safeGet(r,'h1_eval_VAF_shuffs', []);
            p_VAF_eval  = safeGet(r,'h1_eval_VAF_p', NaN);
            med_eval    = safeMed(evalShuffs);
            vecR_shuffs = safeGet(r,'h1_vecR_shuff_all', []);
            p_vecR      = safeGet(r,'h1_vecR_p', NaN);

            boots_inferred = max([numel(refitShuffs), numel(evalShuffs), numel(vecR_shuffs)]);
            if isnan(nBoots), nBoots = boots_inferred; end

            % Flags
            flags = {};
            withinVAF = @(x) all(x(~isnan(x)) > -1 & x(~isnan(x)) < 1.000001);
            if ~(reconId==3), flags{end+1}='reconId!=3'; end
            if ~strcmp(obsMode,expectedMode), flags{end+1}='obsMode!=expected'; end
            if ~(isnan(shufT) || shufT==expectedShufT), flags{end+1}='shuffleTarget!=expected'; end
            if ~withinVAF([VAF_real med_refit med_eval]), flags{end+1}='VAF outside [-1,1]'; end
            if ~isfinite(VAF_real), flags{end+1}='VAF_real NaN'; end
            if boots_inferred==0, flags{end+1}='no shuffles'; end

            % PASS test
            pass_refit = isfinite(VAF_real) && isfinite(med_refit) && (VAF_real > med_refit) && isfinite(p_VAF_refit) && (p_VAF_refit <= 0.05);

            passTxt = 'PASS';
            if ~pass_refit, passTxt='CHECK'; end
            if ~isempty(flags), passTxt=[passTxt ' [' strjoin(flags,',') ']']; end

            % Print line
            if isnan(dataIdx), idxStr='NA'; else, idxStr=sprintf('%d', dataIdx); end
            fprintf('%s\\%s | a%02d s%03d idx=%s | %-7s | VAF=%.3f (ref med=%.3f, p=%.3g) | vecR=%.3f (p=%.3g) | nB=%4d | %s\n',...
                actorFolder, sentFolder, actorID, sentenceID, idxStr, obsMode, ...
                VAF_real, med_refit, p_VAF_refit, vecR_real, p_vecR, nBoots, passTxt);

            % Collect
            rows = [rows; struct( ...
                'actorFolder',actorFolder, 'sentenceFolder',sentFolder, ...
                'folder',subPath, 'mode',obsMode, ...
                'actor',actorID, 'sentence',sentenceID, 'dataIdx',dataIdx, ...
                'VAF',VAF_real, 'VAF_med_refit',med_refit, 'p_VAF_refit',p_VAF_refit, ...
                'VAF_med_eval',med_eval, 'p_VAF_eval',p_VAF_eval, ...
                'vecR',vecR_real, 'p_vecR',p_vecR, ...
                'nBoots',nBoots, 'pass_refit',pass_refit, ...
                'flags',{flags})]; %#ok<AGROW>
        end
    end
end

% Summary
% ===== Extended SUMMARY & ANALYSES =====
if isempty(rows)
    fprintf('\nPresent files: %d | Missing expected files: %d\n', present, missing);
    fprintf('PASS (refit criterion): %d / %d\n', 0, 0);
    summary = rows; 
    return;
end

nPass = sum([rows.pass_refit]); 
nRows = numel(rows);
fprintf('\nPresent files: %d | Missing expected files: %d\n', present, missing);
fprintf('PASS (refit criterion): %d / %d\n', nPass, nRows);

% ---------- Overall mode-wise pass counts (refit & eval-only) ----------
modes = {'MR+VID','MR','VID'};
fprintf('\n=== Overall pass counts by mode ===\n');
for mi = 1:numel(modes)
    m = modes{mi};
    mrows = rows(strcmp({rows.mode}, m));
    pass_refit = sum(isfinite([mrows.p_VAF_refit]) & [mrows.p_VAF_refit] <= 0.05);
    pass_eval  = sum(isfinite([mrows.p_VAF_eval])  & [mrows.p_VAF_eval]  <= 0.05);
    fprintf('%-7s | refit PASS: %2d / %2d | eval-only PASS: %2d / %2d\n', ...
        m, pass_refit, numel(mrows), pass_eval, numel(mrows));
end

% ---------- Primary within-actor summary: ACTOR 08 ----------
mask08 = arrayfun(@(r) isfinite(r.actor) && (r.actor==8), rows);
rows08 = rows(mask08);
fprintf('\n=== Actor 08 (within-actor, across sentences) ===\n');
if isempty(rows08)
    fprintf('No entries for actor 08 found.\n');
else
    for mi = 1:numel(modes)
        m = modes{mi};
        m08 = rows08(strcmp({rows08.mode}, m));
        if isempty(m08)
            fprintf('%-7s | N=0\n', m); 
            continue;
        end
        dVAF = [m08.VAF] - [m08.VAF_med_refit];
        vecR = [m08.vecR];
        pref = [m08.p_VAF_refit];
        pevl = [m08.p_VAF_eval];

        [med_dVAF,q1_dVAF,q3_dVAF,iqr_dVAF] = robust_stats(dVAF);
        [med_vecR,q1_vecR,q3_vecR,iqr_vecR] = robust_stats(vecR);
        med_p_refit = median(pref(isfinite(pref)),'omitnan');
        med_p_eval  = median(pevl(isfinite(pevl)),'omitnan');

        beatShuffle = sum(dVAF > 0);
        pass_refit  = sum(isfinite(pref) & pref <= 0.05);
        N = numel(dVAF);

        fprintf('%-7s | N=%2d | ΔVAF med=%.3f [Q1=%.3f, Q3=%.3f, IQR=%.3f] | vecR med=%.3f [Q1=%.3f, Q3=%.3f, IQR=%.3f] | beat>shuffle: %2d/%2d | refit PASS: %2d/%2d | median p_refit=%.3g | median p_eval=%.3g\n', ...
            m, N, med_dVAF, q1_dVAF, q3_dVAF, iqr_dVAF, ...
            med_vecR, q1_vecR, q3_vecR, iqr_vecR, ...
            beatShuffle, N, pass_refit, N, med_p_refit, med_p_eval);
    end
end

% ---------- Outlier transparency: actor_08 / s005 (if present) ----------
rows0805 = rows(strcmp({rows.actorFolder},'actor_08') & strcmp({rows.sentenceFolder},'s005'));
fprintf('\n=== Outlier check: actor_08\\s005 ===\n');
if isempty(rows0805)
    fprintf('actor_08\\s005 not found (skipping).\n');
else
    for mi = 1:numel(modes)
        m = modes{mi};
        m0805 = rows0805(strcmp({rows0805.mode}, m));
        if isempty(m0805), continue; end
        dVAF = [m0805.VAF] - [m0805.VAF_med_refit];
        pref = [m0805.p_VAF_refit];
        fprintf('%-7s | ΔVAF=%.3f | p_refit=%.3g | VAF=%.3f | null_med=%.3f | vecR=%.3f | p_vecR=%.3g\n', ...
            m, dVAF, pref, m0805.VAF, m0805.VAF_med_refit, m0805.vecR, m0805.p_vecR);
    end

    % Recompute Actor 08 medians excluding s005
    rows08_excl = rows08(~strcmp({rows08.sentenceFolder},'s005'));
    fprintf('\nActor 08 medians (excluding s005):\n');
    for mi = 1:numel(modes)
        m = modes{mi};
        m08x = rows08_excl(strcmp({rows08_excl.mode}, m));
        if isempty(m08x), continue; end
        dVAF = [m08x.VAF] - [m08x.VAF_med_refit];
        vecR = [m08x.vecR];
        [med_dVAF,~,~,iqr_dVAF] = robust_stats(dVAF);
        [med_vecR,~,~,iqr_vecR] = robust_stats(vecR);
        fprintf('%-7s | ΔVAF med=%.3f (IQR=%.3f) | vecR med=%.3f (IQR=%.3f)\n', ...
            m, med_dVAF, iqr_dVAF, med_vecR, iqr_vecR);
    end
end

% ---------- Across-actor exemplars (non-08) ----------
otherActors = unique([rows(~mask08 & arrayfun(@(r) isfinite(r.actor), rows)).actor]);
fprintf('\n=== Across-actor exemplars (each non-08 actor, single sentence items) ===\n');
if isempty(otherActors)
    fprintf('No non-08 actors found.\n');
else
    for a = otherActors
        arows = rows([rows.actor]==a);
        sentFolders = unique({arows.sentenceFolder});
        fprintf('Actor %02d | sentences: %s\n', a, strjoin(sentFolders, ', '));
        for mi = 1:numel(modes)
            m = modes{mi};
            am = arows(strcmp({arows.mode}, m));
            if isempty(am), continue; end
            dVAF = [am.VAF] - [am.VAF_med_refit];
            pref = [am.p_VAF_refit];
            fprintf('  %-7s | items=%d | median ΔVAF=%.3f | median p_refit=%.3g | PASS (p≤.05): %d/%d\n', ...
                m, numel(am), median(dVAF,'omitnan'), median(pref(isfinite(pref)),'omitnan'), ...
                sum(isfinite(pref)&pref<=0.05), numel(pref));
        end
    end
end

% ---------- Global mode-wise medians across all items ----------
modes = {'MR+VID','MR','VID'};
fprintf('\n=== Global mode-wise medians across all items ===\n');
for mi = 1:numel(modes)
    m = modes{mi};
    mrows = rows(strcmp({rows.mode}, m));
    if isempty(mrows)
        fprintf('%-7s | N=0\n', m); continue;
    end
    dVAF = [mrows.VAF] - [mrows.VAF_med_refit];
    vecR = [mrows.vecR];
    pref = [mrows.p_VAF_refit];
    pevl = [mrows.p_VAF_eval];
    [med_dVAF,q1_dVAF,q3_dVAF,iqr_dVAF] = robust_stats(dVAF);
    [med_vecR,q1_vecR,q3_vecR,iqr_vecR] = robust_stats(vecR);
    beatShuffle = sum(dVAF > 0);
    pass_refit  = sum(isfinite(pref) & pref<=0.05);
    pass_eval   = sum(isfinite(pevl) & pevl<=0.05);
    fprintf('%-7s | N=%2d | ΔVAF med=%.3f [Q1=%.3f,Q3=%.3f,IQR=%.3f] | vecR med=%.3f [Q1=%.3f,Q3=%.3f,IQR=%.3f] | beat>shuffle: %2d/%2d | refit PASS: %2d/%2d | eval PASS: %2d/%2d\n',...
        m, numel(mrows), med_dVAF, q1_dVAF, q3_dVAF, iqr_dVAF, ...
        med_vecR, q1_vecR, q3_vecR, iqr_vecR, ...
        beatShuffle, numel(mrows), pass_refit, numel(mrows), pass_eval, numel(mrows));
end


% ---------- With vs without outlier (actor_08\s005) ----------
outMask = strcmp({rows.actorFolder},'actor_08') & strcmp({rows.sentenceFolder},'s005');
rows_wo = rows(~outMask);

fprintf('\n=== Global medians WITH vs WITHOUT actor_08\\s005 ===\n');
for mi = 1:numel(modes)
    m = modes{mi};
    m_all = rows(strcmp({rows.mode}, m));
    m_wo  = rows_wo(strcmp({rows_wo.mode}, m));
    if isempty(m_all), continue; end
    d_all = [m_all.VAF] - [m_all.VAF_med_refit];
    d_wo  = [m_wo.VAF]  - [m_wo.VAF_med_refit];
    [med_all,~,~,iqr_all] = robust_stats(d_all);
    [med_wo, ~,~,iqr_wo ] = robust_stats(d_wo);
    fprintf('%-7s | ΔVAF med (ALL)=%.3f (IQR=%.3f)  vs  (WO)=%.3f (IQR=%.3f) | N_all=%d, N_wo=%d\n',...
        m, med_all, iqr_all, med_wo, iqr_wo, numel(d_all), numel(d_wo));
end

% Actor 08 only: with vs without s005 (re-compute mask)
rows08 = rows(arrayfun(@(r) isfinite(r.actor) && r.actor==8, rows));
rows08_wo = rows08(~strcmp({rows08.sentenceFolder},'s005'));

fprintf('\n=== Actor 08 medians WITH vs WITHOUT s005 ===\n');
for mi = 1:numel(modes)
    m = modes{mi};
    m08   = rows08(strcmp({rows08.mode}, m));
    m08wo = rows08_wo(strcmp({rows08_wo.mode}, m));
    if isempty(m08), continue; end
    d08   = [m08.VAF]   - [m08.VAF_med_refit];
    d08wo = [m08wo.VAF] - [m08wo.VAF_med_refit];
    v08   = [m08.vecR];   v08wo = [m08wo.vecR];
    [med_d,~,~,iqr_d]    = robust_stats(d08);
    [med_dwo,~,~,iqr_dwo]= robust_stats(d08wo);
    [med_v,~,~,iqr_v]    = robust_stats(v08);
    [med_vwo,~,~,iqr_vwo]= robust_stats(v08wo);
    fprintf('%-7s | ΔVAF med ALL=%.3f (IQR=%.3f) vs WO=%.3f (IQR=%.3f) | vecR med ALL=%.3f (IQR=%.3f) vs WO=%.3f (IQR=%.3f)\n',...
        m, med_d, iqr_d, med_dwo, iqr_dwo, med_v, iqr_v, med_vwo, iqr_vwo);
end

% ---------- Speaker-balanced summary: per-actor medians & sign test ----------
fprintf('\n=== Speaker-balanced summary (per-actor medians, by mode) ===\n');
for mi = 1:numel(modes)
    m = modes{mi};
    mrows = rows(strcmp({rows.mode}, m));
    if isempty(mrows), fprintf('%-7s | N=0\n', m); continue; end
    % keep entries with a finite actor id
    mrows = mrows(arrayfun(@(r) isfinite(r.actor), mrows));
    if isempty(mrows), fprintf('%-7s | N=0\n', m); continue; end

    actors = unique([mrows.actor]);
    med_dVAF_perActor = nan(size(actors));
    med_vecR_perActor = nan(size(actors));
    for i = 1:numel(actors)
        ai = actors(i);
        ar = mrows([mrows.actor] == ai);
        med_dVAF_perActor(i) = median([ar.VAF] - [ar.VAF_med_refit], 'omitnan');
        med_vecR_perActor(i) = median([ar.vecR], 'omitnan');
    end

    ci_d = boot_ci_median(med_dVAF_perActor, 5000);
    ci_r = boot_ci_median(med_vecR_perActor, 5000);
    p_sign_d = sign_test_right(med_dVAF_perActor);   % H1: ΔVAF > 0 ?
    p_sign_r = sign_test_right(med_vecR_perActor);   % (optional) vec-r > 0 ?

    txt = strjoin(arrayfun(@(a,md,vr)sprintf('a%02d:ΔVAF=%.3f,vecR=%.3f', ...
                        a, md, vr), actors, med_dVAF_perActor, med_vecR_perActor, 'uni', 0), '; ');
    fprintf('%-7s | actors=%d | med(ΔVAF)=%.3f [%.3f, %.3f], sign p=%.3g | med(vecR)=%.3f [%.3f, %.3f], sign p=%.3g\n      %s\n', ...
        m, numel(actors), median(med_dVAF_perActor,'omitnan'), ci_d(1), ci_d(2), p_sign_d, ...
        median(med_vecR_perActor,'omitnan'), ci_r(1), ci_r(2), p_sign_r, txt);
end


% ---------- H1: Matched sentence across speakers (explicit list; MR+VID→Audio) ----------
% Phrase: "Miss black thought about the lap" (sen_252)
fprintf('\n=== H1: Matched sentence across speakers (MR+VID → Audio) ===\n');
files_xspeaker = {
    'C:\Users\jaker\Research-Project\data\results\H1\actor_08\s001\H1_MR+VID.mat', 'a08 s001 — sub8_sen_252_18';
    'C:\Users\jaker\Research-Project\data\results\H1\actor_01\s001\H1_MR+VID.mat', 'a01 s001 — sub1_sen_252_1';
    'C:\Users\jaker\Research-Project\data\results\H1\actor_14\s001\H1_MR+VID.mat', 'a14 s001 — sub14_sen_252_14';
};

for i = 1:size(files_xspeaker,1)
    f = files_xspeaker{i,1};
    lab = files_xspeaker{i,2};
    if ~isfile(f), fprintf('%s: MISSING (%s)\n', lab, f); continue; end

    S = load(f);
    % pick result struct (your H1 saves either as r1 or results)
    if isfield(S,'r1'), R = S.r1;
    elseif isfield(S,'results'), R = S.results;
    else, R = struct(); 
    end

    % pull metrics from the actual H1 fields
    VAF_real = NaN; if isfield(R,'h1_VAF_real'), VAF_real = R.h1_VAF_real; end
    vecR     = NaN; if isfield(R,'h1_vecR_real'), vecR     = R.h1_vecR_real; end
    refitSh  = [];  if isfield(R,'h1_refit_VAF_shuffs'), refitSh = R.h1_refit_VAF_shuffs; end
    VAF_med_refit = NaN; if ~isempty(refitSh), VAF_med_refit = median(refitSh,'omitnan'); end

    dVAF = VAF_real - VAF_med_refit;
    fprintf('%s | ΔVAF=%.3f | vecR=%.3f | VAF=%.3f | ref_med=%.3f\n', lab, dVAF, vecR, VAF_real, VAF_med_refit);
end


% ---------- Top & Bottom 3 items by ΔVAF per mode ----------
fprintf('\n=== Extremes by ΔVAF (Top 3 / Bottom 3) per mode ===\n');
for mi = 1:numel(modes)
    m = modes{mi};
    mrows = rows(strcmp({rows.mode}, m));
    if isempty(mrows), continue; end
    dVAF = [mrows.VAF] - [mrows.VAF_med_refit];
    [d_sorted, idx] = sort(dVAF, 'descend');
    kTop = min(3, numel(d_sorted));
    kBot = min(3, numel(d_sorted));
    fprintf('\n%-7s | TOP %d by ΔVAF\n', m, kTop);
    for k = 1:kTop
        r = mrows(idx(k));
        fprintf('  %s\\%s | a%02d s%03d | ΔVAF=%.3f | p_refit=%.3g | vecR=%.3f\n',...
            r.actorFolder, r.sentenceFolder, r.actor, r.sentence, d_sorted(k), r.p_VAF_refit, r.vecR);
    end
    fprintf('%-7s | BOTTOM %d by ΔVAF\n', m, kBot);
    for k = 1:kBot
        r = mrows(idx(end-k+1));
        fprintf('  %s\\%s | a%02d s%03d | ΔVAF=%.3f | p_refit=%.3g | vecR=%.3f\n',...
            r.actorFolder, r.sentenceFolder, r.actor, r.sentence, dVAF(idx(end-k+1)), r.p_VAF_refit, r.vecR);
    end
end



% Return raw rows for further programmatic use
summary = rows;

end

% --- helpers ---
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

function [med,q1,q3,iqrW] = robust_stats(x)
    x = x(isfinite(x));
    if isempty(x)
        med = NaN; q1 = NaN; q3 = NaN; iqrW = NaN;
        return;
    end
    med = median(x);
    q = prctile(x,[25 75]);
    q1 = q(1); q3 = q(2);
    iqrW = q3 - q1;
end

function ci = boot_ci_median(x, B)
    x = x(~isnan(x));
    if numel(x) < 2, ci = [NaN NaN]; return; end
    if nargin < 2, B = 5000; end
    n = numel(x); meds = zeros(B,1);
    for b = 1:B
        idx = randi(n, n, 1);
        meds(b) = median(x(idx));
    end
    ci = quantile(meds, [0.025 0.975]);
end

function p = sign_test_right(d)
    % One-sided sign test for median(d) > 0
    d = d(~isnan(d));
    d = d(d~=0);  % drop zeros per sign-test convention
    n = numel(d);
    if n == 0, p = NaN; return; end
    k = sum(d > 0);
    p = binocdf(k-1, n, 0.5, 'upper');
end




analyse_h1_results('C:\Users\jaker\Research-Project\data\results');
