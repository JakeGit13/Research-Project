function generate_loadings_scatter(results,i)

    subplot(1,3,i);   % <— i will be passed in
    plot(results.nonShuffledLoadings,results.nonShuffledReconLoadings,'.');

    
    % Unity line
    hline=refline(1,0);
    
    hline.Color = 'k';
    xlabel('Original loadings');ylabel('Reconstructed loadings');
    


end


% H1 scatter triptych — a08 s003 (MR+VID, MR, VID)
folder = 'C:\Users\jaker\Research-Project\data\results\H1\actor_08\s003';

files = {
    fullfile(folder, 'H1_MR+VID.mat'), 'MR+VID';
    fullfile(folder, 'H1_MR_ONLY.mat'), 'MR';
    fullfile(folder, 'H1_VID_ONLY.mat'), 'VID';
};



figure;  % open a single figure before the loop
for i = 1:3
    % load results struct (r1 or results)
    S = load(files{i,1});
    if isfield(S,'r1'), R = S.r1; else, R = S.results; end

    generate_loadings_scatter(R,i);
end
sgtitle('H1 — a08 s003 | Loadings scatter by observed mode', 'Interpreter','none');


%% Difficulty spectrum — MR+VID only (actor_08: s003 PASS, s006 borderline, s005 outlier)
figure;  % new figure for the 3 MR+VID panels

files_diff = {
    'C:\Users\jaker\Research-Project\data\results\H1\actor_08\s003\H1_MR+VID.mat', 'a08 s003 (PASS)';   % think these need to be more informative 
    'C:\Users\jaker\Research-Project\data\results\H1\actor_08\s006\H1_MR+VID.mat', 'a08 s006 (borderline)';
    'C:\Users\jaker\Research-Project\data\results\H1\actor_08\s005\H1_MR+VID.mat', 'a08 s005 (outlier)';
};

for i = 1:3
    S = load(files_diff{i,1});
    if isfield(S,'r1'), R = S.r1; else, R = S.results; end
    generate_loadings_scatter(R, i);              % uses subplot(1,3,i)
    title(sprintf('MR+VID \x2192 Audio | %s', files_diff{i,2}), 'Interpreter','none');
end

sgtitle('H1 — MR+VID to Audio | actor 08 difficulty spectrum', 'Interpreter','none');




%% Cross-speaker — MR+VID only, same sentence: "Miss black thought about the lap"
figure;  % new figure for 3 MR+VID panels across actors

files_xspeaker = {
    'C:\Users\jaker\Research-Project\data\results\H1\actor_08\s001\H1_MR+VID.mat', 'a08 s001 — sub8\_sen\_252\_18';
    'C:\Users\jaker\Research-Project\data\results\H1\actor_01\s001\H1_MR+VID.mat', 'a01 s001 — sub1\_sen\_252\_1';
    'C:\Users\jaker\Research-Project\data\results\H1\actor_14\s001\H1_MR+VID.mat', 'a14 s001 — sub14\_sen\_252\_14';
};

for i = 1:3
    actorNum = {'8','1','14'};
    S = load(files_xspeaker{i,1});
    if isfield(S,'r1'), R = S.r1; else, R = S.results; end
    generate_loadings_scatter(R, i);  % uses subplot(1,3,i)
    title(sprintf('Actor %s' actorNum, 'Interpreter','none');
end

sgtitle('MR+VID to Audio | same sentence across 3 actors ', ...
        'Interpreter','none');



% === H1 global summary: ΔVAF across items by mode (single figure) ===
h1Root = "C:\Users\jaker\Research-Project\data\results\H1";

modes = {'MR+VID','MR','VID'};
files = {'H1_MR+VID.mat','H1_MR.mat','H1_VID.mat'};
allDelta = cell(1,3);

actors = dir(fullfile(h1Root,'actor_*')); 
actors = actors([actors.isdir]);

for a = 1:numel(actors)
    sents = dir(fullfile(h1Root, actors(a).name, 's*'));
    sents = sents([sents.isdir]);
    for s = 1:numel(sents)
        sd = fullfile(h1Root, actors(a).name, sents(s).name);
        for m = 1:3
            fp = fullfile(sd, files{m});
            if exist(fp,'file')
                S = load(fp);
                d = NaN;
                % try common field names
                if isfield(S,'deltaVAF'), d = S.deltaVAF; end
                if isnan(d) && isfield(S,'dVAF'), d = S.dVAF; end
                % try inside a struct (e.g., results/r1)
                if isnan(d)
                    fn = fieldnames(S);
                    for k = 1:numel(fn)
                        v = S.(fn{k});
                        if isstruct(v)
                            if isfield(v,'deltaVAF'), d = v.deltaVAF; break; end
                            if isfield(v,'dVAF'), d = v.dVAF; break; end
                            if isfield(v,'VAF') && isfield(v,'null_med')
                                d = v.VAF - v.null_med; break;
                            end
                        end
                    end
                end
                % fallback: top-level VAF-null_med
                if isnan(d) && isfield(S,'VAF') && isfield(S,'null_med')
                    d = S.VAF - S.null_med;
                end
                if ~isnan(d), allDelta{m}(end+1) = d; end %#ok<AGROW>
            end
        end
    end
end

% --- plot: jittered dots per mode with median line ---
figure('Name','H1 ΔVAF by mode','Color','w'); hold on;
for m = 1:3
    y = allDelta{m}(:);
    x = m + 0.1*randn(size(y));           % light jitter
    plot(x, y, 'k.', 'MarkerSize', 10);   % dots
    med = median(y, 'omitnan');
    plot([m-0.25 m+0.25], [med med], 'r-', 'LineWidth', 2);  % median bar
end
yline(0,'k-'); 
xlim([0.5 3.5]);
set(gca,'XTick',1:3,'XTickLabel',modes);
ylabel('ΔVAF (model − shuffle median)'); 
title('Audio reconstruction (H1): ΔVAF across items by mode');


% Optional save:
% outdir = fullfile('C:\Users\jaker\Research-Project\data\results','figures','H1');
% if ~exist(outdir,'dir'), mkdir(outdir); end
% base = 'H1_MR+VID_sameSentence_crossActors';
% savefig(gcf, fullfile(outdir, [base '.fig']));
% exportgraphics(gcf, fullfile(outdir, [base '.pdf']), 'ContentType','vector');
% exportgraphics(gcf, fullfile(outdir, [base '.png']), 'Resolution', 400);

