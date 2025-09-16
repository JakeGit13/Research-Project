function generate_loadings_scatter(results,i)

    subplot(1,3,i);   % <— i will be passed in
    plot(results.nonShuffledLoadings,results.nonShuffledReconLoadings,'.');

    
    % Unity line
    hline=refline(1,0);
    
    hline.Color = 'k';
    xlabel('Original loadings');ylabel('Reconstructed loadings');
    

    %{
    
    figure; % Doesn't work as we don't have shuffleStats
    
    % Original and reconstructed loadings
    subplot(2,3,2);
    plot(results.nonShuffledLoadings,results.nonShuffledReconLoadings,'.');
    
    % Unity line
    hline=refline(1,0);
    hline.Color = 'k';
    
    xlabel('Original loadings');ylabel('Reconstructed loadings');
    statStrings = {'Correlation coefficient','Linear Fit Gradient','SSE'};
    
    for statI = 1:3
        % Shuffled distributions    
        subplot(2,3,statI+3);    
        histogram(shuffstats(statI,:),50);hold on    
        axis tight  
        plot(unshuffstats([statI statI]),ylim,'r--','linewidth',2);
        xlabel(statStrings{statI});ylabel('Frequency');
    end

    %}

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
    S = load(files_xspeaker{i,1});
    if isfield(S,'r1'), R = S.r1; else, R = S.results; end
    generate_loadings_scatter(R, i);  % uses subplot(1,3,i)
    title(sprintf('MR+VID \x2192 Audio | %s', files_xspeaker{i,2}), 'Interpreter','none');
end

sgtitle('H1 — MR+VID \rightarrow Audio | same sentence across actors: "Miss black thought about the lap"', ...
        'Interpreter','none');

% Optional save:
% outdir = fullfile('C:\Users\jaker\Research-Project\data\results','figures','H1');
% if ~exist(outdir,'dir'), mkdir(outdir); end
% base = 'H1_MR+VID_sameSentence_crossActors';
% savefig(gcf, fullfile(outdir, [base '.fig']));
% exportgraphics(gcf, fullfile(outdir, [base '.pdf']), 'ContentType','vector');
% exportgraphics(gcf, fullfile(outdir, [base '.png']), 'Resolution', 400);

