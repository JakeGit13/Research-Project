function analyse_h2_results(resultsDir)
% Core H2 analysis (no plotting, no file outputs).
% Reads h2_bimodal_results.csv and h2_trimodal_results.csv,
% pairs rows by (data_idx, reconstruct_id), computes deltas,
% and prints summaries overall, by target, and by audio share (if present).

    if nargin < 1 || isempty(resultsDir)
        error('Provide resultsDir (folder containing the H2 CSVs).');
    end
    fBi  = fullfile(resultsDir, 'h2_bimodal_results.csv');
    fTri = fullfile(resultsDir, 'h2_trimodal_results.csv');
    if ~exist(fBi,'file') || ~exist(fTri,'file')
        error('Missing H2 CSVs in %s', resultsDir);
    end

    Tbi  = readtable(fBi);
    Ttri = readtable(fTri);

    % Keep necessary columns only (robust to extras/missing)
    needBi  = intersect({'data_idx','reconstruct_id','native_R','native_R2','native_SSE'}, Tbi.Properties.VariableNames);
    needTri = intersect({'data_idx','reconstruct_id','native_R','native_R2','native_SSE','include_audio','target_audio_share'}, Ttri.Properties.VariableNames);
    Tbi  = Tbi(:, needBi);
    Ttri = Ttri(:, needTri);

    % Trimodal rows only (if the flag exists)
    if any(strcmp('include_audio', Ttri.Properties.VariableNames))
        Ttri = Ttri(Ttri.include_audio==1, :);
    end

    % Pair: (data_idx, reconstruct_id)
    J = innerjoin(Ttri, Tbi, 'Keys', {'data_idx','reconstruct_id'}, ...
        'RightVariables', {'native_R','native_R2','native_SSE'});

    % Rename paired columns for clarity
    J.Properties.VariableNames = strrep(J.Properties.VariableNames,'native_R_Tbi','native_R_bi');
    J.Properties.VariableNames = strrep(J.Properties.VariableNames,'native_R2_Tbi','native_R2_bi');
    J.Properties.VariableNames = strrep(J.Properties.VariableNames,'native_SSE_Tbi','native_SSE_bi');
    J.Properties.VariableNames = strrep(J.Properties.VariableNames,'native_R_Ttri','native_R_tri');
    J.Properties.VariableNames = strrep(J.Properties.VariableNames,'native_R2_Ttri','native_R2_tri');
    J.Properties.VariableNames = strrep(J.Properties.VariableNames,'native_SSE_Ttri','native_SSE_tri');

    if isempty(J)
        fprintf('No paired H2 rows found. Nothing to analyse.\n');
        return;
    end

    % Deltas (trimodal - bimodal); want ΔR2>0 and ΔSSE>0
    J.delta_R2  = J.native_R2_tri - J.native_R2_bi;
    J.delta_R   = J.native_R_tri  - J.native_R_bi;
    J.delta_SSE = J.native_SSE_bi - J.native_SSE_tri;

    fprintf('Paired H2 rows: %d\n', height(J));

    % Overall summary
    summarizeBlock('Overall', J.delta_R2, J.delta_SSE);

    % By target (1=MR, 2=Video)
    for rid = intersect([1 2], unique(J.reconstruct_id(:)')) 
        m = (J.reconstruct_id==rid);
        label = ternary(rid==1,'Target=MR','Target=Video');
        summarizeBlock(label, J.delta_R2(m), J.delta_SSE(m));

        % Paired tests (right-tailed: tri > bi)
        dR2 = J.delta_R2(m); dR2 = dR2(isfinite(dR2));
        if ~isempty(dR2)
            [~,p_t,~,st] = ttest(dR2, 0, 'Tail','right');
            d_cohen = mean(dR2) / max(std(dR2), eps);
            try
                p_w = signrank(dR2, 0, 'tail','right');
            catch, p_w = NaN;
            end
            fprintf('  Stats: t-test p=%.4g (t=%.3f, df=%d), signrank p=%.4g, Cohen d=%.3f\n', ...
                p_t, st.tstat, st.df, p_w, d_cohen);
        end
    end

    % By audio share (if present)
    if any(strcmp('target_audio_share', J.Properties.VariableNames))
        shares = unique(J.target_audio_share(~isnan(J.target_audio_share))).';
        for s = shares
            m = (J.target_audio_share==s);
            summarizeBlock(sprintf('Audio share=%.3f', s), J.delta_R2(m), J.delta_SSE(m));
        end
    end
end

function summarizeBlock(name, dR2, dSSE)
    dR2  = dR2(isfinite(dR2));
    dSSE = dSSE(isfinite(dSSE));
    if isempty(dR2)
        fprintf('[%s] no data\n', name); 
        return;
    end
    q = quantile(dR2, [0.25 0.5 0.75]);
    win = mean(dR2 > 0);
    fprintf('\n[%s]\n', name);
    fprintf('  ΔR²:  mean=%.4f  median=%.4f  IQR=[%.4f, %.4f]  (%%>0 = %.1f%%)\n', ...
        mean(dR2), q(2), q(1), q(3), 100*win);
    if ~isempty(dSSE)
        qS = quantile(dSSE, [0.25 0.5 0.75]);
        fprintf('  ΔSSE: mean=%.4f  median=%.4f  IQR=[%.4f, %.4f]\n', ...
            mean(dSSE), qS(2), qS(1), qS(3));
    end
end

function out = ternary(cond,a,b)

    if cond
        out=a; 
    else,
        out=b; 
    end
end


resultsRoot = 'C:\Users\jaker\Research-Project\data\results'
analyse_h2_results(resultsRoot);