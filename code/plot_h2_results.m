function plot_h2_results

clc;
h2Root = "C:\Users\jaker\Research-Project\data\results\H2";


sentList = {
    'actor_08','sub8_sen_257_15_svtimriMANUAL','VID';  % VID max-gain
    'actor_14','sub14_sen_252_14_svtimriMANUAL','MR';  % MR max-gain
    % 'actor_08','sub1_sen_252_1_svtimriMANUAL','VID'  % OPTIONAL: paper anchor
};

% === Tri exemplars: original vs reconstructed loadings (per sentence) ===
sentList = {
    'actor_08','sub8_sen_257_15_svtimriMANUAL','VID';  % VID max-gain
    'actor_14','sub14_sen_252_14_svtimriMANUAL','MR';  % MR max-gain
    % 'actor_08','sub1_sen_252_1_svtimriMANUAL','VID'  % OPTIONAL: paper anchor
};

for i = 1:size(sentList,1)
    actorName = sentList{i,1};
    sentName  = sentList{i,2};
    targetStr = sentList{i,3}; % 'VID' or 'MR'

    sentDir = fullfile(h2Root, actorName, sentName);
    mf = dir(fullfile(sentDir, ['H2_target' targetStr '*.mat']));
    if isempty(mf), warning('No H2_target%s file in %s', targetStr, sentDir); continue; end

    S = load(fullfile(sentDir, mf(1).name));

    % find a struct with the loadings we need
    if isfield(S,'results')
        results = S.results;
    else
        results = [];
        fn = fieldnames(S);
        for j = 1:numel(fn)
            v = S.(fn{j});
            if isstruct(v) && all(isfield(v,{'nonShuffledLoadings','nonShuffledReconLoadings'}))
                results = v; break;
            end
        end
        if isempty(results), warning('No results struct with required fields in %s', mf(1).name); continue; end
    end

    orig = results.nonShuffledLoadings(:);
    reco = results.nonShuffledReconLoadings(:);

    lims = [min([orig; reco]) max([orig; reco])];
    if ~any(isfinite(lims)), warning('Non-finite values in %s', mf(1).name); continue; end
    r = range(lims); lims = lims + 0.05*[-1 1]*r;

    figure('Name',sprintf('Tri scatter — %s | %s | %s',targetStr,actorName,sentName),'Color','w');
    plot(orig, reco, '.'); hold on;
    plot(lims, lims, 'k-');
    hline=refline(1,0); hline.Color = 'k';
    axis equal; xlim(lims); ylim(lims);
    R = corr(orig, reco, 'rows','complete');
    p = polyfit(orig, reco, 1); slope = p(1);
    SSE = sum((reco - orig).^2);
    xlabel('Original loadings'); ylabel('Reconstructed loadings');
    title(sprintf('%s | %s | %s', targetStr, actorName, sentName), 'Interpreter','none');
    text(0.05,0.95,sprintf('R=%.3f, slope=%.2f, SSE=%.2g', R, slope, SSE),...
         'Units','normalized','VerticalAlignment','top');
end



% --- pick one actor
actors = dir(fullfile(h2Root,'actor_*')); 
actors = actors([actors.isdir]);
actorDir = fullfile(h2Root, actors(1).name);     % change index if needed

% --- pick one sentence under that actor
sentences = dir(fullfile(actorDir,'sub*_sen_*')); 
sentences = sentences([sentences.isdir]);
sentDir = fullfile(actorDir, sentences(1).name); % change index if needed

% --- find the H2 results file in that sentence folder
matFiles = dir(fullfile(sentDir,'H2_*.mat'));
if isempty(matFiles)
    error('No H2 results files found in %s', sentDir);
end
% prefer targetVID if present (else take the first file)
pick = 1;
for k = 1:numel(matFiles)
    if contains(matFiles(k).name,'targetVID','IgnoreCase',true)
        pick = k; break;
    end
end
dataFile = fullfile(sentDir, matFiles(pick).name);

% --- show what's inside (helps confirm field names)
info = whos('-file', dataFile);
fprintf('Loading: %s\nVariables: %s\n', dataFile, strjoin({info.name}, ', '));

% --- load variables
S = load(dataFile);

% results: find a struct that has the two loadings we need
if isfield(S,'results')
    results = S.results;
else
    results = [];
    fn = fieldnames(S);
    for i = 1:numel(fn)
        val = S.(fn{i});
        if isstruct(val) && all(isfield(val, {'nonShuffledLoadings','nonShuffledReconLoadings'}))
            results = val; break;
        end
    end
    if isempty(results)
        error('Could not find a struct with nonShuffledLoadings and nonShuffledReconLoadings in %s', dataFile);
    end
end

% shuffle stats (optional)
shuffstats = [];
unshuffstats = [];
if isfield(S,'shuffstats') && isfield(S,'unshuffstats')
    shuffstats = S.shuffstats;
    unshuffstats = S.unshuffstats;
elseif isfield(results,'shuffstats') && isfield(results,'unshuffstats')
    shuffstats = results.shuffstats;
    unshuffstats = results.unshuffstats;
end

fprintf('=== Plotting actor: %s | sentence: %s ===\n', actors(1).name, sentences(1).name);

% -------------------------------
% Your existing plotting code
% -------------------------------
figure;
plot(results.nonShuffledLoadings, results.nonShuffledReconLoadings, '.');
hline = refline(1,0); hline.Color = 'k';
xlabel('Original loadings'); ylabel('Reconstructed loadings'); title(matFiles(pick).name, 'Interpreter','none');

figure;
subplot(2,3,2);
plot(results.nonShuffledLoadings, results.nonShuffledReconLoadings, '.');
hline = refline(1,0); hline.Color = 'k';
xlabel('Original loadings'); ylabel('Reconstructed loadings'); title(matFiles(pick).name, 'Interpreter','none');

statStrings = {'Correlation coefficient','Linear Fit Gradient','SSE'};
if ~isempty(shuffstats) && ~isempty(unshuffstats)
    for statI = 1:3
        subplot(2,3,statI+3);
        histogram(shuffstats(statI,:),50); hold on; axis tight
        plot(unshuffstats([statI statI]), ylim, 'r--','linewidth',2);
        xlabel(statStrings{statI}); ylabel('Frequency');
    end
else
    % no shuffle stats in this file — skip histograms
    subplot(2,3,4:6);
    axis off;
    text(0.5,0.5,'No shuffle stats found in this file.\nSkipping histograms.',...
         'HorizontalAlignment','center','VerticalAlignment','middle');
end


%{

% Suppose you loaded:
% resultsBi.nonShuffledLoadings, resultsBi.nonShuffledReconLoadings
% resultsTri.nonShuffledLoadings, resultsTri.nonShuffledReconLoadings

orig = resultsBi.nonShuffledLoadings(:);
recoBi = resultsBi.nonShuffledReconLoadings(:);
recoTri = resultsTri.nonShuffledReconLoadings(:);

lims = [min([orig; recoBi; recoTri]) max([orig; recoBi; recoTri])];
lims = lims + 0.05*[-1 1]*range(lims);

figure('Name','H2 Sentence Exemplar'); 

% --- Bi panel
subplot(1,2,1);
plot(orig, recoBi, '.'); hold on;
plot(lims, lims, 'k-'); axis equal; xlim(lims); ylim(lims);
R = corr(orig, recoBi, 'rows','complete');
p = polyfit(orig, recoBi, 1); slope = p(1);
SSE = sum((recoBi - orig).^2);
xlabel('Original loadings'); ylabel('Reconstructed loadings'); title('Bi');
text(0.05,0.95,sprintf('R=%.3f, slope=%.2f, SSE=%.2g', R, slope, SSE),...
     'Units','normalized','VerticalAlignment','top');

% --- Tri panel
subplot(1,2,2);
plot(orig, recoTri, '.'); hold on;
plot(lims, lims, 'k-'); axis equal; xlim(lims); ylim(lims);
R = corr(orig, recoTri, 'rows','complete');
p = polyfit(orig, recoTri, 1); slope = p(1);
SSE = sum((recoTri - orig).^2);
xlabel('Original loadings'); ylabel('Reconstructed loadings'); title('Tri');
text(0.05,0.95,sprintf('R=%.3f, slope=%.2f, SSE=%.2g', R, slope, SSE),...
     'Units','normalized','VerticalAlignment','top');

%}

% --- Build pooled R-vectors for the group plot (BiR/TriR for MR & VID) ---
BiR_MR = []; TriR_MR = [];
BiR_VID = []; TriR_VID = [];



function plot_h2_delta_waterfall(BiR, TriR, actorIDs, labels, titleStr)
% Sorted ΔR per item; color by actor.
dR = TriR(:) - BiR(:);
valid = ~isnan(dR);
dR = dR(valid); actorIDs = actorIDs(valid); labels = labels(valid);

% sort by ΔR desc
[~, order] = sort(dR, 'descend');
dR = dR(order); actorIDs = actorIDs(order); labels = labels(order);

figure('Name',['H2 ΔR — ' titleStr],'Color','w'); 
bar(dR, 'FaceColor', [0.6 0.6 0.6]); hold on;
yline(0, 'k-'); % zero line
ylim([min(0,min(dR)-0.01) max(dR)+0.01]);
xlim([0.5 numel(dR)+0.5]);
xlabel('Items (sorted by ΔR)'); ylabel('ΔR = TriR − BiR'); title(titleStr);

% color-code by actor (simple: color bars by actor ID)
uActs = unique(actorIDs(~isnan(actorIDs)));
colors = lines(numel(uActs));
for i = 1:numel(dR)
    a = actorIDs(i);
    if ~isnan(a)
        c = colors(uActs==a, :);
        set(get(gca,'Children'),'FaceColor','flat'); % ensure FaceColor supports CData
        b = findobj(gca,'Type','bar'); % last bar handle
        b.CData(i,:) = c;
    end
end
% annotate median ΔR
medDelta = median(dR);
plot(xlim, [medDelta medDelta], 'r--', 'LineWidth', 1);
text(0.02, 0.95, sprintf('median ΔR=%.3f', medDelta), 'Units','normalized', 'Color','r', 'VerticalAlignment','top');
end



% From analyse_h2 you should have item-aligned arrays:
% mr_Rbi, mr_Rtri, mr_actor, mr_labels   and   vid_Rbi, vid_Rtri, vid_actor, vid_labels

plot_h2_delta_waterfall(mr_Rbi,  mr_Rtri,  mr_actor,  mr_labels,  'MR target');
plot_h2_delta_waterfall(vid_Rbi, vid_Rtri, vid_actor, vid_labels, 'VID target');




end
