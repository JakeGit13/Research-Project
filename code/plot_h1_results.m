function generate_loadings_scatter(results)

    figure;
    
    % Original and reconstructed loadings
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


a_08_s002_MR = load('C:\Users\jaker\Research-Project\data\results\H1\actor_08\s002\H1_MR_ONLY.mat');

generate_loadings_scatter(a_08_s002_MR.r1);

