function plot_h2_results(h2Root, saveDir)
% H2 core plots (across items only):
% (1) R(Bi) vs R(Tri) scatter   (MR, VID)
% (2) ΔR distribution           (MR, VID)
% (3) Baseline dependence       (R(Bi) vs ΔR) with Spearman ρ (+ bootstrap CI)

if nargin<1 || isempty(h2Root), h2Root = 'C:\Users\jaker\Research-Project\data\results\H2'; end
if ~isfolder(h2Root), error('Not found: %s', h2Root); end
if nargin<2, saveDir = ''; end
if ~isempty(saveDir) && ~isfolder(saveDir), mkdir(saveDir); end

set(groot,'defaultFigureColor','w'); set(groot,'defaultAxesBox','off'); % simple H1-like style

% ---------- collect ----------
[mr, vid] = collectH2(h2Root);

% ---------- (1) R(Bi) vs R(Tri) ----------
fig1 = figure('Name','H2_Rbi_vs_Rtri'); tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; scatterBiTri(mr, 'Target=MR');
nexttile; scatterBiTri(vid,'Target=VID');
saveIf(fig1, saveDir, 'H2_Rbi_vs_Rtri.png');

% ---------- (2) ΔR distribution ----------
fig2 = figure('Name','H2_dR_distribution'); tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; distDeltaR(mr, 'ΔR (Target=MR)');
nexttile; distDeltaR(vid,'ΔR (Target=VID)');
saveIf(fig2, saveDir, 'H2_dR_distribution.png');

% ---------- (3) Baseline dependence ----------
fig3 = figure('Name','H2_baseline_dependence'); tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; baselineDep(mr, 'Baseline dependence (MR)');
nexttile; baselineDep(vid,'Baseline dependence (VID)');
saveIf(fig3, saveDir, 'H2_baseline_dependence.png');

fprintf('Done.\n');

% ================= helpers =================
function [mr, vid] = collectH2(root)
    mr  = struct('Rbi',[],'Rtri',[],'dR',[]);
    vid = struct('Rbi',[],'Rtri',[],'dR',[]);
    actors = dir(fullfile(root,'actor_*')); actors = actors([actors.isdir]);
    for a = 1:numel(actors)
        items = dir(fullfile(actors(a).folder, actors(a).name));
        items = items([items.isdir]); items = items(~ismember({items.name},{'.','..'}));
        for i = 1:numel(items)
            base = fullfile(actors(a).folder, actors(a).name, items(i).name);
            add(fullfile(base,'H2_targetMR_shufAUD.mat'), mr);
            add(fullfile(base,'H2_targetVID_shufAUD.mat'), vid);
        end
    end
    function add(fp, tgt)
        if ~exist(fp,'file'), return; end
        S = load(fp); Rtri=NaN; Rbi=NaN;
        fns = fieldnames(S);
        for k=1:numel(fns)
            v = S.(fns{k});
            if isstruct(v) && isfield(v,'h2_tri') && isfield(v,'h2_bi')
                if isfield(v.h2_tri,'R'), Rtri = v.h2_tri.R; end
                if isfield(v.h2_bi,'R'),  Rbi  = v.h2_bi.R;  end
                break;
            end
        end
        if ~isnan(Rtri) && ~isnan(Rbi)
            tgt.Rbi(end+1)  = Rbi; %#ok<AGROW>
            tgt.Rtri(end+1) = Rtri; %#ok<AGROW>
            tgt.dR(end+1)   = Rtri - Rbi; %#ok<AGROW>
        end
    end
end

function scatterBiTri(d, ttl)
    x = d.Rbi(:); y = d.Rtri(:); if isempty(x), title(ttl); return; end
    plot(x,y,'ko','MarkerSize',4,'MarkerFaceColor','k'); hold on;
    lims = [min([x;y])*0.98, 1]; plot(lims,lims,'k:'); axis square; xlim(lims); ylim(lims);
    xlabel('R (Bi: MR+VID)'); ylabel('R (Tri: MR+VID+AUD)'); title(ttl);
    dR = y - x; med = median(dR,'omitnan'); pct = 100*mean(y>x);
    text(lims(1)+0.03*range(lims), lims(2)-0.07*range(lims), ...
        sprintf('median \\DeltaR = %.3f   |   %% above y=x = %.1f%%', med, pct), ...
        'FontSize',9,'BackgroundColor','w','Margin',3,'EdgeColor',[0.85 0.85 0.85]);
    hold off;
end

function distDeltaR(d, ttl)
    v = d.dR(:); if isempty(v), title(ttl); return; end
    xj = 1 + 0.06*(rand(size(v))-0.5);
    plot(xj, v,'ko','MarkerSize',4,'MarkerFaceColor','k'); hold on;
    med = median(v,'omitnan'); ci = bootCIMedian(v, 5000);
    plot([0.85 1.15],[med med],'k-','LineWidth',1.5);
    plot([1 1],ci,'k-','LineWidth',3);
    xlim([0.75 1.25]); xticks([]); ylabel('\DeltaR = R(Tri) - R(Bi)'); title(ttl);
    text(0.77, max(ylim)-0.06*range(ylim), ...
        sprintf('median = %.3f  [%.3f, %.3f]', med, ci(1), ci(2)), ...
        'FontSize',9,'BackgroundColor','w','Margin',3,'EdgeColor',[0.85 0.85 0.85]);
    hold off;
end

function baselineDep(d, ttl)
    x = d.Rbi(:); y = d.dR(:); if isempty(x), title(ttl); return; end
    plot(x,y,'ko','MarkerSize',4,'MarkerFaceColor','k'); hold on;
    ok = ~isnan(x) & ~isnan(y); if nnz(ok)>=2
        p = polyfit(x(ok),y(ok),1); xx = linspace(min(x(ok)),max(x(ok)),100); plot(xx,polyval(p,xx),'k-');
    end
    [rho,pv] = corr(x,y,'Type','Spearman','Rows','complete');
    ci = bootCISpearman(x,y,3000);
    xlabel('R (Bi)'); ylabel('\DeltaR'); title(ttl); axis square;
    text(min(xlim)+0.03*range(xlim), max(ylim)-0.07*range(ylim), ...
        sprintf('\\rho_S = %.3f  [%.3f, %.3f],  p=%.3g', rho, ci(1), ci(2), pv), ...
        'FontSize',9,'BackgroundColor','w','Margin',3,'EdgeColor',[0.85 0.85 0.85]);
    hold off;
end

function ci = bootCIMedian(x,B)
    x = x(~isnan(x)); n=numel(x); if n<2, ci=[NaN NaN]; return; end
    if nargin<2, B=5000; end
    m=zeros(B,1); for b=1:B, m(b)=median(x(randi(n,n,1))); end
    ci = quantile(m,[0.025 0.975]);
end

function ci = bootCISpearman(x,y,B)
    ok = ~isnan(x) & ~isnan(y); x=x(ok); y=y(ok); n=numel(x); if n<3, ci=[NaN NaN]; return; end
    if nargin<3, B=3000; end
    r=zeros(B,1); for b=1:B, idx=randi(n,n,1); r(b)=corr(x(idx),y(idx),'Type','Spearman'); end
    ci = quantile(r,[0.025 0.975]);
end

function saveIf(fig, outDir, name)
    if isempty(outDir), return; end
    exportgraphics(fig, fullfile(outDir,name), 'Resolution', 200);
end
end
