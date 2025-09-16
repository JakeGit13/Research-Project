function generate_loadings_scatter(results)

    % --- recompute unshuffstats and shuffstats if missing ---
    x0 = results.nonShuffledLoadings(:);
    y0 = results.nonShuffledReconLoadings(:);

    % Unshuffled stats
    R0 = corr(x0,y0,'rows','pairwise');
    p  = polyfit(x0,y0,1);
    slope0 = p(1);
    yhat   = polyval(p,x0);
    SSE0   = sum((y0 - yhat).^2);
    unshuffstats = [R0; slope0; SSE0];

    % Shuffled stats (3 × nShuffles)
    if isfield(results,'allShuffledOrigLoad') && isfield(results,'allShuffledReconLoad')
        Xsh = results.allShuffledOrigLoad;
        Ysh = results.allShuffledReconLoad;

        % ensure correct shape [nLoadings × nShuffles]
        if size(Xsh,1) ~= numel(x0)
            Xsh = Xsh.'; Ysh = Ysh.'; % transpose if needed
        end

        nSh = size(Xsh,2);
        shuffstats = nan(3,nSh);
        for s = 1:nSh
            xs = Xsh(:,s); ys = Ysh(:,s);
            Rs = corr(xs,ys,'rows','pairwise');
            p  = polyfit(xs,ys,1);
            slope = p(1);
            yhat  = polyval(p,xs);
            SSE   = sum((ys - yhat).^2);
            shuffstats(:,s) = [Rs; slope; SSE];
        end
    else
        shuffstats = [];
    end

    % --- plot scatter (Fig A) ---
    figure;
    plot(x0,y0,'.');
    hline=refline(1,0); hline.Color = 'k';
    xlabel('Original loadings'); ylabel('Reconstructed loadings');

    % --- composite figure (Fig B) ---
    figure;
    subplot(2,3,2);
    plot(x0,y0,'.');
    hline=refline(1,0); hline.Color = 'k';
    xlabel('Original loadings'); ylabel('Reconstructed loadings');

    statStrings = {'Correlation coefficient','Linear Fit Gradient','SSE'};

    for statI = 1:3
        subplot(2,3,statI+3);
        if ~isempty(shuffstats)
            histogram(shuffstats(statI,:),50); hold on; axis tight;
            yl = ylim;
            plot([unshuffstats(statI) unshuffstats(statI)], yl, 'r--','linewidth',2);
            xlabel(statStrings{statI}); ylabel('Frequency');
        else
            text(0.5,0.5,'No shuffled stats','HorizontalAlignment','center');
            axis off;
        end
    end

end


a_08_s002_MR = load('C:\Users\jaker\Research-Project\data\results\H1\actor_08\s002\H1_MR_ONLY.mat');
generate_loadings_scatter(a_08_s002_MR.r1);
