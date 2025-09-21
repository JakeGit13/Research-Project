function analyse_H2_results(h2Root)
% Super short H2 summary: Tri vs Bi R with ΔR. No analysis, just prints.

clc;

actors = dir(fullfile(h2Root,'actor_*')); actors = actors([actors.isdir]);
fprintf('=== H2 summary ===\nRoot: %s\n', h2Root);

% Collectors for minimal + extra H2 stats
mr_Rtri = []; mr_Rbi = []; mr_dR = []; mr_pTriR = []; mr_actor = []; mr_labels = {};
vid_Rtri = []; vid_Rbi = []; vid_dR = []; vid_pTriR = []; vid_actor = []; vid_labels = {};



for a = 1:numel(actors)
    aPath = fullfile(actors(a).folder, actors(a).name);
    items = dir(aPath); items = items([items.isdir]);
    items = items(~ismember({items.name},{'.','..'}));
    if isempty(items), continue; end

    fprintf('\n%s\n', upper(actors(a).name));
    for i = 1:numel(items)
        iPath = fullfile(aPath, items(i).name);

        f = fullfile(iPath,'H2_targetMR_shufAUD.mat');
        if exist(f,'file'); printLine(items(i).name, 'MR', f); end

        f = fullfile(iPath,'H2_targetVID_shufAUD.mat');
        if exist(f,'file'); printLine(items(i).name, 'VID', f); end
    end
end

% ------- Compact group summaries (per target) -------
printSummary('MR',  mr_dR,  mr_pTriR);
printSummary('VID', vid_dR, vid_pTriR);

% ------- Extra analysis #1: baseline dependence (Spearman, with CI) -------
printBaselineDependence('MR',  mr_Rbi,  mr_dR);
printBaselineDependence('VID', vid_Rbi, vid_dR);

% ------- Extra analysis #2: actor-level inference (median per actor) -------
printActorSummary('MR',  mr_actor,  mr_dR);
printActorSummary('VID', vid_actor, vid_dR);


fprintf('\nDone.\n');

function printLine(itemName, tgt, fp)
    S = load(fp);
    r2 = pickR2(S);
    meta = pickMeta(S);

    Rtri = getR(r2,'h2_tri'); Rbi = getR(r2,'h2_bi');
    if isnan(Rtri) || isnan(Rbi), return; end
    dR = getDelta(r2,'dR', Rtri - Rbi);
    pTriR = getP(r2,'h2_p','R');  % Tri vs shuffle p (may be NaN)
    sigStr = ''; if ~isnan(pTriR) && pTriR<0.05, sigStr=' *'; end

    fprintf('  %-20s | %-3s  TriR=%.3f  BiR=%.3f  ΔR=%.3f%s\n', itemName, tgt, Rtri, Rbi, dR, sigStr);

    actorID = NaN; if ~isempty(meta) && isfield(meta,'actorID'), actorID = meta.actorID; end
    switch upper(tgt)
        case 'MR'
            mr_Rtri(end+1) = Rtri; %#ok<AGROW>
            mr_Rbi(end+1)  = Rbi;  %#ok<AGROW>
            mr_dR(end+1)   = dR;   %#ok<AGROW>
            mr_pTriR(end+1)= pTriR;%#ok<AGROW>
            mr_actor(end+1)= actorID; %#ok<AGROW>
            mr_labels{end+1} = itemName; %#ok<AGROW>
        case 'VID'
            vid_Rtri(end+1) = Rtri; %#ok<AGROW>
            vid_Rbi(end+1)  = Rbi;  %#ok<AGROW>
            vid_dR(end+1)   = dR;   %#ok<AGROW>
            vid_pTriR(end+1)= pTriR;%#ok<AGROW>
            vid_actor(end+1)= actorID; %#ok<AGROW>
            vid_labels{end+1} = itemName; %#ok<AGROW>
    end
end



function r2 = pickR2(S)
    r2 = struct(); fn = fieldnames(S);
    for k=1:numel(fn)
        v = S.(fn{k});
        if isstruct(v) && isfield(v,'h2_tri') && isfield(v,'h2_bi'), r2=v; return; end
    end
    if ~isempty(fn) && isstruct(S.(fn{1})), r2 = S.(fn{1}); end
end

function r = getR(r2,field)
    if isfield(r2,field) && isfield(r2.(field),'R'), r=r2.(field).R; else, r=NaN; end
end

function d = getDelta(r2,name,fallback)
    if isfield(r2,'h2_delta') && isfield(r2.h2_delta,name)
        d = r2.h2_delta.(name);
    else
        d = fallback;
    end
end

function p = getP(r2, field, subfield)
    if isfield(r2, field) && isfield(r2.(field), subfield)
        p = r2.(field).(subfield);
    else
        p = NaN;
    end
end

function printSummary(label, dR, pTriR)
    dR = dR(~isnan(dR));
    if isempty(dR)
        fprintf('\n=== H2 SUMMARY (%s) ===\nNo items.\n', label);
        return;
    end
    n = numel(dR);
    med = median(dR);
    ci = boot_ci_median(dR, 5000);            % 95% bootstrap CI of median
    p_sign = sign_test_right(dR);             % one-sided: ΔR>0 ?
    pctPos = 100*mean(dR>0);

    fracShuffle = NaN; nAvail = sum(~isnan(pTriR));
    if nAvail > 0
        fracShuffle = 100*mean(pTriR(~isnan(pTriR)) < 0.05);
    end

    fprintf('\n=== H2 SUMMARY (%s) ===\n', label);
    fprintf('Items=%d | median ΔR=%.3f [%.3f, %.3f] | %%ΔR>0=%.1f%% | sign-test p=%.3g\n', ...
        n, med, ci(1), ci(2), pctPos, p_sign);
    if ~isnan(fracShuffle)
        fprintf('Tri vs shuffle p<0.05: %.1f%% of items (%d/%d)\n', fracShuffle, round(fracShuffle*nAvail/100), nAvail);
    end
end

function ci = boot_ci_median(x, B)
    x = x(~isnan(x));
    if numel(x) < 2, ci = [NaN NaN]; return; end
    if nargin<2, B = 5000; end
    n = numel(x);
    meds = zeros(B,1);
    for b = 1:B
        idx = randi(n, n, 1);
        meds(b) = median(x(idx));
    end
    ci = quantile(meds, [0.025 0.975]);
end

function p = sign_test_right(d)
    % One-sided sign test for median(d) > 0
    d = d(~isnan(d));
    d = d(d~=0);                 % drop zeros per sign test convention
    n = numel(d);
    if n == 0, p = NaN; return; end
    k = sum(d > 0);
    % P(X >= k) for X~Binom(n,0.5)
    p = binocdf(k-1, n, 0.5, 'upper');
end

function meta = pickMeta(S)
    meta = [];
    fn = fieldnames(S);
    for k=1:numel(fn)
        v = S.(fn{k});
        if isstruct(v) && any(isfield(v, {'actorID','wavName','dataIdx'}))
            meta = v; return;
        end
    end
end

function printBaselineDependence(label, Rbi, dR)
    mask = ~(isnan(Rbi) | isnan(dR));
    Rbi = Rbi(mask); dR = dR(mask);
    if numel(Rbi) < 3
        fprintf('\nBaseline dependence (%s): insufficient items.\n', label); return;
    end
    [rho,p] = corr(Rbi(:), dR(:), 'Type','Spearman','Rows','complete');
    ci = boot_ci_spearman(Rbi(:), dR(:), 5000);
    fprintf('Baseline dep. (%s): Spearman rho=%.3f [%.3f, %.3f], p=%.3g, n=%d\n', ...
        label, rho, ci(1), ci(2), p, numel(Rbi));
end

function ci = boot_ci_spearman(x,y,B)
    if nargin<3, B=5000; end
    n = numel(x); rhos = zeros(B,1);
    for b=1:B
        idx = randi(n,n,1);
        rhos(b) = corr(x(idx), y(idx), 'Type','Spearman');
    end
    ci = quantile(rhos,[0.025 0.975]);
end

function printActorSummary(label, actorVec, dR)
    ok = ~isnan(actorVec) & ~isnan(dR);
    if ~any(ok)
        fprintf('Actor-level (%s): no actor IDs found.\n', label); return;
    end
    actors = unique(actorVec(ok));
    if isempty(actors)
        fprintf('Actor-level (%s): no actor IDs found.\n', label); return;
    end
    medPerActor = nan(size(actors));
    for i=1:numel(actors)
        medPerActor(i) = median(dR(actorVec==actors(i)));
    end
    ci = boot_ci_median(medPerActor, 5000);
    p = sign_test_right(medPerActor);
    txt = strjoin(arrayfun(@(a,m)sprintf('actor_%02d=%.3f',a,m), actors, medPerActor,'uni',0), ', ');
    fprintf('Actor-level (%s): medΔR per actor: %s | group median=%.3f [%.3f, %.3f] | sign-test p=%.3g (n=%d actors)\n', ...
        label, txt, median(medPerActor), ci(1), ci(2), p, numel(actors));
end



end


analyse_H2_results("C:\Users\jaker\Research-Project\data\results\H2");