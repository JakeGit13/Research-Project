function plot_h2_results

clc;
h2Root = "C:\Users\jaker\Research-Project\data\results\H2";

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


end
