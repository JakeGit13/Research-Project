function analyse_H1_results(resultsDir)
% Core H1 analysis (no plotting, no file outputs).
% Expects h1_audio_results.csv with columns:
%   data_idx, reconstruct_id==3, source_label in {'MR+Video','MR','Video'},
%   audio_R_native, audio_R2_native, audio_SSE_native, (optional) p_SSE_audio, target_audio_share, nBoots

    if nargin < 1 || isempty(resultsDir)
        error('Provide resultsDir (folder containing the H1 CSV).');
    end
    fH1 = fullfile(resultsDir, 'h1_audio_results.csv');
    if ~exist(fH1,'file'), error('Missing %s', fH1); end

    T = readtable(fH1);

    % --- Normalise source_label names to a consistent set ---
    if any(strcmp('source_label', T.Properties.VariableNames))
        T.source_label = strrep(T.source_label, 'MRVID', 'MR+Video');
        T.source_label = strrep(T.source_label, 'VID',   'Video');
        % keep 'MR' as is
    end


    % keep only audio-target rows if others slipped in
    if any(strcmp('reconstruct_id', T.Properties.VariableNames))
        T = T(T.reconstruct_id==3, :);
    end

    % --- per-source summaries ---
    srcs = {'MR+Video','MR','Video'};
    fprintf('H1: per-source summaries (audio_R2_native)\n');
    for s = srcs
        m = strcmpi(T.source_label, s{1}) & isfinite(T.audio_R2_native);
        if ~any(m), fprintf('  %s: no data\n', s{1}); continue; end
        x = T.audio_R2_native(m);
        q = quantile(x,[.25 .5 .75]);
        fprintf('  %-8s: mean=%.4f  median=%.4f  IQR=[%.4f, %.4f]  n=%d\n', ...
            s{1}, mean(x), q(2), q(1), q(3), nnz(m));

        % Fisher combined p (if p_SSE_audio present & finite)
        if any(strcmp('p_SSE_audio', T.Properties.VariableNames))
            p = T.p_SSE_audio(m);
            p = p(isfinite(p) & p>0 & p<=1);
            if ~isempty(p)
                X2 = -2*sum(log(p));
                k  = 2*numel(p);
                pF = 1 - chi2cdf(X2, k);
                fprintf('       Fisher combined p (SSE null) = %.3g (n=%d)\n', pF, numel(p));
            end
        end
    end

    % --- pairwise, sentence-level comparisons: MR+Video vs MR / Video ---
    % reshape wide: one row per data_idx with columns for each source
    varsKeep = intersect({'data_idx','source_label','audio_R2_native'}, T.Properties.VariableNames);
    W = unstack(T(:,varsKeep), 'audio_R2_native', 'source_label');  % columns: MR, MR+Video, Video (as available)

    % Helper to print a paired summary
    function paired_summary(label, tri, bi)
        m = isfinite(tri) & isfinite(bi);
        d = tri(m) - bi(m);
        if ~any(m)
            fprintf('%s: no paired data\n', label); return;
        end
        q = quantile(d,[.25 .5 .75]);
        [~,p_t,~,st] = ttest(d, 0, 'Tail','right');     % test MR+Vid > comparator
        try, p_w = signrank(d, 0, 'tail','right'); catch, p_w = NaN; end
        d_cohen = mean(d)/max(std(d),eps);
        fprintf('\n%s (paired across sentences):\n', label);
        fprintf('  ΔR² mean=%.4f  median=%.4f  IQR=[%.4f, %.4f]  (%%>0=%.1f%%)  n=%d\n', ...
            mean(d), q(2), q(1), q(3), 100*mean(d>0), nnz(m));
        fprintf('  Stats: t-test p=%.4g (t=%.3f, df=%d), signrank p=%.4g, Cohen d=%.3f\n', ...
            p_t, st.tstat, st.df, p_w, d_cohen);
    end

    % MR+Video vs MR
    if all(ismember({'MR+Video','MR'}, W.Properties.VariableNames))
        paired_summary('MR+Video vs MR', W.("MR+Video"), W.MR);
    end
    % MR+Video vs Video
    if all(ismember({'MR+Video','Video'}, W.Properties.VariableNames))
        paired_summary('MR+Video vs Video', W.("MR+Video"), W.Video);
    end

    % (optional) quick note on nBoots granularity if present
    if any(strcmp('nBoots', T.Properties.VariableNames))
        nb = unique(T.nBoots(isfinite(T.nBoots)));
        if ~isempty(nb)
            fprintf('\nNote: H1 bootstrap granularity p_min = 1/(nBoots+1); observed nBoots: %s\n', mat2str(nb(:)'));
        end
    end
end


resultsRoot = 'C:\Users\jaker\Research-Project\data\results';
analyse_H1_results(resultsRoot);