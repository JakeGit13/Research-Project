function audioBlock = extractAudioFeatures(processedAudio, opts)

    arguments
        processedAudio 
        opts.VERBOSE  = true
        opts.useNoiseAudio = false
        opts.genFigures
    end

    VERBOSE = opts.VERBOSE;
    useNoiseAudio = opts.useNoiseAudio;
    genFigures = opts.genFigures;

    % --- pull from preprocessing struct ---
    frameSnippets = processedAudio.frameSnippets;   % [winLen x nFrames]
    sampleRate    = processedAudio.sampleRate;
    [winLen, nFrames] = size(frameSnippets);
    
    fprintf('[INFO] fs=%g Hz | window=%d samples (%.1f ms) | frames=%d\n', ...
            sampleRate, winLen, 1000*winLen/sampleRate, nFrames);
    


    %% === Features 1 & 2: log-F0 and voicing probability ===
    % Input: frameSnippets [winLen x nFrames], sampleRate
    % Output rows (so far): [logF0 ; vuvProb]
    
    % Parameters (local defaults; can be overridden by processedAudio.*)
    minF0 = 50;            % Hz
    maxF0 = 300;           % Hz
    unvoicedValue = 0;     % value used for logF0 when unvoiced
    vuvThresh = 0.40;      % NACF threshold for voicing (0..1)
    
    if isfield(processedAudio,'minF0'),         minF0 = processedAudio.minF0;         end
    if isfield(processedAudio,'maxF0'),         maxF0 = processedAudio.maxF0;         end
    if isfield(processedAudio,'unvoicedValue'), unvoicedValue = processedAudio.unvoicedValue; end
    if isfield(processedAudio,'vuvThresh'),     vuvThresh = processedAudio.vuvThresh; end
    
    % Derived settings
    minLag = max(1, floor(sampleRate / maxF0));          % samples
    maxLag = min(winLen-1, ceil(sampleRate / minF0));    % limit to window length
    if maxLag <= minLag
        warning('F0 lag range invalid for this window. Adjusting to [minLag=2, maxLag=%d].', max(minLag+1,2));
        minLag = 2; maxLag = max(minLag+1, 3);
    end
    nfft = 2^nextpow2(2*winLen);                          % for FFT-based autocorrelation
    
    if opts.VERBOSE
        fprintf('[PARAM] F0 range=%d–%d Hz | unvoicedValue=%g | vuvThresh=%.2f | lags=%d–%d | nfft=%d\n', ...
            minF0, maxF0, unvoicedValue, vuvThresh, minLag, maxLag, nfft);
    end


    % --- HF ratio helpers (single-sided spectrum indices) ---
    half_n = floor(nfft/2) + 1;
    freqs  = (0:half_n-1) * (sampleRate / nfft);
    nyq    = sampleRate / 2;
    
    hfLoHz = 4000;                       % default lower bound of HF band (Hz)
    if hfLoHz >= nyq
        hfLoHz = max(0.5*nyq, nyq-1);    % keep inside Nyquist if fs is low
    end
    hf_mask = (freqs >= hfLoHz) & (freqs <= 0.98*nyq);
    if ~any(hf_mask)
        hf_mask = freqs >= 0.5*nyq;      % fallback: top half of band
    end







    %% --- High-band mel filterbank (for melPC1) ---
    mel   = @(f) 2595*log10(1 + f/700);
    invmel= @(m) 700*(10.^(m/2595) - 1);
    
    f_lo = max(4000, freqs(2));      % start ≥ 4 kHz, clamp to >0
    f_hi = 0.98 * nyq;               % up to ~Nyquist
    
    if f_hi <= f_lo
        % Fallback if sampling rate is too low: take top quarter band
        f_lo = 0.75*nyq;
        f_hi = 0.98*nyq;
    end
    
    nMel = 20;                        % number of triangular mel bands
    m_edges = linspace(mel(f_lo), mel(f_hi), nMel+2);
    f_edges = invmel(m_edges);
    
    % Triangular filters over single-sided bin freqs
    M = zeros(nMel, half_n);
    for b = 1:nMel
        fL = f_edges(b); fC = f_edges(b+1); fR = f_edges(b+2);
        % left slope
        L = (freqs >= fL) & (freqs <= fC);
        % right slope
        R = (freqs >= fC) & (freqs <= fR);
        M(b,L) = (freqs(L) - fL) / max(fC - fL, eps);
        M(b,R) = (fR - freqs(R)) / max(fR - fC, eps);
    end
    
    if opts.VERBOSE
        fprintf('[MEL] high band: %.0f–%.0f Hz | nBands=%d\n', f_lo, f_hi, nMel);
    end

    % Preallocate
    f0_hz   = zeros(1, nFrames);
    vuvProb = zeros(1, nFrames);
    cpp_db  = zeros(1, nFrames); 
    hf_ratio = zeros(1, nFrames);
    mel_logE = zeros(nMel, nFrames);   % per-frame high-band log-mel energies

    
    % Loop frames
    for t = 1:nFrames
        x = double(frameSnippets(:,t));
        x = x - mean(x);                        % remove DC
    
        if ~any(x)                              % silence window
            vuvProb(t) = 0; 
            f0_hz(t)   = 0;
            cpp_db(t)  = 0;
            hf_ratio(t) = 0;
            continue
        end

    
        % Autocorrelation via Wiener–Khinchin
        X  = fft(x, nfft);
        r  = ifft(abs(X).^2, 'symmetric');      % autocorrelation
        r0 = max(r(1), eps);
        r  = r ./ r0;                           % normalized ACF


        % --- HF energy ratio: sum(|X|) in high band / sum(|X|) overall (single-sided) ---
        mag = abs(X(1:half_n));           % single-sided magnitude
        E_all = sum(mag);
        E_hi  = sum(mag(hf_mask));
        hf_ratio(t) = E_hi / max(E_all, eps);

        % --- High-band log-mel energies (single-sided power spectrum) ---
        pow = (abs(X(1:half_n))).^2;
        melE = M * pow(:);                 % [nMel x 1]
        mel_logE(:,t) = log(melE + eps);   % natural log






        %% --- CPP: cepstral peak prominence over the pitch quefrency band ---
        log_mag = log(abs(X) + eps);
        log_mag = log_mag - mean(log_mag);              % remove DC in log-spectrum
        cep = real(ifft(log_mag));                      % real cepstrum
        
        qIdx = minLag:maxLag;                           % pitch-relevant quefrency range
        [cpeak, cposRel] = max(cep(qIdx));              % peak height in that band
        cpos = qIdx(1) + cposRel - 1;                   % absolute index of the peak
        
        % Simple baseline: linear trend over the band, evaluated at peak location
        p_lin = polyfit(qIdx, double(cep(qIdx)).', 1);  % degree-1 fit
        cbase = polyval(p_lin, cpos);
        
        % Convert natural-log difference to dB-like scale (20/log(10)≈8.6859)
        cpp_db(t) = 8.685889638 * (cpeak - cbase);

    
        % Search peak in allowed lag range
        rSeg = r(minLag:maxLag);
        [peak, idx] = max(rSeg);
        lag = minLag + idx - 1;
        candF0 = sampleRate / lag;
    
        % Save voicing probability (peak NACF in 0..1)
        vuvProb(t) = max(min(peak, 1), 0);
    
        % Accept candidate only if in-range; else leave as 0 (unvoiced)
        if candF0 >= minF0 && candF0 <= maxF0
            f0_hz(t) = candF0;
        else
            vuvProb(t) = 0;  % unreliable peak → treat as unvoiced
        end
    end
    
    % Map to final features
    vuvFlag = double(vuvProb >= vuvThresh);     % optional binary flag (not returned)
    % Robust lf0 with interpolation over unvoiced frames
    voiced = (f0_hz > 0) & (vuvFlag == 1);              % logical 1×T
    logF0  = nan(1, nFrames, 'like', f0_hz);            % start with NaNs
    
    logF0(voiced) = log(max(f0_hz(voiced), eps));       % natural log
    
    % Interpolate NaNs across time; then fill any leading/trailing NaNs
    logF0 = fillmissing(logF0, 'linear');
    logF0 = fillmissing(logF0, 'nearest');
    
    % Fallback if no voiced frames at all
    if all(~voiced)
        logF0(:) = 0;  % neutral fallback; keeps downstream stable
    end

    




    %% === High-band mel PC1 across time (frames) ===
    Y = mel_logE.';                          % [frames x nMel]
    Yc = bsxfun(@minus, Y, mean(Y,1));       % center variables (bands)
    [U,S,~] = svd(Yc, 'econ');               % PCA via SVD
    melPC1 = (U(:,1) * S(1,1)).';            % [1 x nFrames] first PC scores
    
    if opts.VERBOSE
        svals = diag(S);
        expl1 = (svals(1)^2) / max(sum(svals.^2), eps);
        fprintf('[MEL PC1] variance explained = %.1f%%\n', 100*expl1);
    end

   


    %% VERBOSE SUMMARY PRINTS (need to add the rest)
    % Verbose summary (F0)
    if opts.VERBOSE
        nVoiced = nnz(voiced);
        fprintf('[F0] Range=%.0f–%.0f Hz | lag=%d–%d | voiced %d/%d (%.1f%%)\n', ...
            minF0, maxF0, minLag, maxLag, nVoiced, nFrames, 100*nVoiced/nFrames);
        if nVoiced > 0
            fprintf('[F0] Median F0=%.1f Hz | Mean NACF(voiced)=%.2f\n', ...
                median(f0_hz(voiced)), mean(vuvProb(voiced)));
        end


        % Verbose summary (CPP)
        validCPP = isfinite(cpp_db);
        if any(validCPP)
            fprintf('[CPP] Median=%.2f dB | Mean=%.2f dB\n', ...
                median(cpp_db(validCPP)), mean(cpp_db(validCPP)));
        else
            fprintf('[CPP] No valid values.\n');
        end


        % Verbose summary (HF)
        fprintf('[HF ratio] band = %.0f–%.0f Hz | median=%.3f | mean=%.3f\n', ...
            freqs(find(hf_mask,1,'first')), freqs(find(hf_mask,1,'last')), ...
            median(hf_ratio), mean(hf_ratio));
    end




    




    %% Assemble features into output block (features x frames)
    audioBlock = [logF0; vuvProb; cpp_db; hf_ratio; melPC1];  % full set


    %% Low-pass smooth features along time to match MR/Video bandwidth
    % Set fps to your per-frame rate (e.g., 16.7 or 25)
    fps  = 16.7;                     % Estimate based on paper 
    cut  = 8;                        % Hz cutoff (6–7 Hz recommended)
    win  = max(1, round(fps / cut)); % moving-average window (frames)
    
    audioBlock = movmean(audioBlock, win, 2, 'Endpoints', 'shrink');




    %% CALL PLOTTING FUNCTIONS
    if genFigures 

        % plot_F0_voicing(processedAudio, audioBlock) 

        % plot_CPP(processedAudio, audioBlock);

        % plot_HF_ratio(processedAudio, audioBlock);

        plot_melPC1(processedAudio, audioBlock);



        
    end




    %% Optional: RETURN noise block instead of real features
    % useNoiseAudio = isfield(processedAudio,'useNoiseAudio') && logical(processedAudio.useNoiseAudio);
    if useNoiseAudio
        rng(42);  
        audioBlock = randn(size(audioBlock), 'double');   % same dims: 5 x nFrames
    end
    
    if opts.VERBOSE
        tag = ternary(useNoiseAudio, 'NOISE', 'REAL');
        fprintf('[OUT %s] features x frames = %d x %d\n', tag, size(audioBlock,1), size(audioBlock,2));
    end
    
    




   

end

% tiny helper (local ternary) GET RID OF THIS 
function y = ternary(cond, a, b)
    
    if cond
        y=a; 
    else
        y=b; 
    end
end



%% PLOTTING FUNCTIONS
function plot_F0_voicing(processedAudio, audioBlock)
% Single figure:
%   - Background: frame spectrogram (right y-axis in Hz)
%   - Overlay: voicing probability (left y-axis 0..1) as solid black line
%   - Red dashed horizontal line at voicing threshold (default 0.40)

    % ---- Inputs ----
    X  = processedAudio.frameSnippets;   % [winLen x nFrames]
    fs = processedAudio.sampleRate;
    if isfield(processedAudio,'vuvThresh'), vuvThresh = processedAudio.vuvThresh; else, vuvThresh = 0.40; end

    vuvProb = audioBlock(2,:);           % voicing probability

    % ---- Axes/time ----
    [winLen, nFrames] = size(X);
    t = (0:nFrames-1) * (winLen/fs);     % seconds

    % ---- Per-frame spectrogram ----
    nfft   = 2^nextpow2(2*winLen);
    halfN  = floor(nfft/2)+1;
    specDB = zeros(halfN, nFrames);
    for k = 1:nFrames
        xk = X(:,k) - mean(X(:,k));      % DC remove
        mag = abs(fft(xk, nfft));
        specDB(:,k) = 20*log10(mag(1:halfN) + eps);
    end
    f = (0:halfN-1) * (fs/nfft);         % Hz
    climHi = prctile(specDB(:),95);
    climLo = climHi - 60;
    yMaxHz = min(4000, fs/2);            % compact speech band

    % ---- Figure with two overlaid axes ----
    figure('Color','w');

    % Bottom axis: spectrogram (right y-axis)
    axSpec = axes('Position',[0.12 0.15 0.78 0.75]); %#ok<LAXES>
    imagesc(axSpec, t, f, specDB); axis(axSpec, 'xy');
    colormap(axSpec, parula); caxis(axSpec, [climLo climHi]);
    ylim(axSpec, [0, yMaxHz]);
    xlim(axSpec, [t(1) t(end)]);
    ylabel(axSpec, 'Frequency (Hz)');
    set(axSpec,'YAxisLocation','right');
    title(axSpec, 'Spectrogram (Hz) with voicing probability (0-1) and threshold');

    % Top axis: voicing probability (transparent, left y-axis)
    axVUV = axes('Position', axSpec.Position, 'Color','none', 'YAxisLocation','left', ...
                 'XLim', axSpec.XLim, 'YLim', [0 1], 'XTickLabel',[]);
    hold(axVUV, 'on');
    plot(axVUV, t, vuvProb, 'k-', 'LineWidth', 2.0);
    yline(axVUV, vuvThresh, '--r', 'LineWidth', 2.0);
    ylabel(axVUV, 'Voicing probability');
    linkaxes([axSpec, axVUV],'x');
    uistack(axVUV, 'top');   % ensure the line stays above the image

    % Shared X label on bottom axis
    xlabel(axSpec, 'Time (s)');
end

function plot_CPP(processedAudio, audioBlock)
% One-figure CPP visualisation:
%  - CPP(dB) over time (left y) + voicing probability (right y) with threshold
%
% Inputs
%  processedAudio.frameSnippets : [winLen x nFrames]  (used only for timing)
%  processedAudio.sampleRate    : scalar
%  processedAudio.vuvThresh     : optional (default 0.40)
%  audioBlock rows: [1]=logF0, [2]=vuvProb, [3]=cpp_db

    % ---- Inputs / defaults ----
    X  = processedAudio.frameSnippets;
    fs = processedAudio.sampleRate;
    if isfield(processedAudio,'vuvThresh'), vuvThresh = processedAudio.vuvThresh; else, vuvThresh = 0.40; end

    vuvProb = audioBlock(2,:);          % 0..1
    cpp_db  = audioBlock(3,:);          % dB

    % ---- Time axis from frames ----
    [winLen, nFrames] = size(X);
    t = (0:nFrames-1) * (winLen/fs);    % seconds
    tlim = [t(1) t(end)];

    % ---- Figure ----
    figure('Color','w');

    % CPP over time + voicing probability
    yyaxis left
    plot(t, cpp_db, 'k-', 'LineWidth', 1.5); grid on;
    ylabel('CPP (dB)');

    % sensible CPP y-limits
    finiteCPP = cpp_db(isfinite(cpp_db));
    if isempty(finiteCPP), finiteCPP = 0; end
    ypad = 0.2 * max(1, iqr(finiteCPP));
    ylim([min(finiteCPP)-ypad, max(finiteCPP)+ypad]);

    yyaxis right
    plot(t, vuvProb, 'Color',[0.1 0.1 0.9], 'LineWidth', 1.0); hold on;
    yline(vuvThresh, '--r', 'LineWidth', 1.0);
    ylim([0 1]); ylabel('Voicing probability');

    xlim(tlim);
    xlabel('Time (s)');
    title('CPP over time (blue) with voicing probability (black)');

end

function plot_HF_ratio(processedAudio, audioBlock, frameIdx)

% Visualise the high-frequency energy ratio (hf_ratio)
% Top: spectrum of one frame with HF band shaded
% Bottom: hf_ratio time series (0..1)
%
% Inputs:
%   processedAudio.frameSnippets : [winLen x nFrames]
%   processedAudio.sampleRate    : scalar
%   audioBlock(4,:)              : hf_ratio per frame (0..1)
%   frameIdx (optional)          : frame to show in the spectrum panel
%
% Usage:
%   plot_HF_ratio(processedAudio, audioBlock);            % auto-pick max hf_ratio frame
%   plot_HF_ratio(processedAudio, audioBlock, 15);        % show frame 15

    % --- Inputs ---
    X  = processedAudio.frameSnippets;
    fs = processedAudio.sampleRate;
    hf_ratio = audioBlock(4,:);            % 0..1

    [winLen, nFrames] = size(X);
    t = (0:nFrames-1) * (winLen/fs);       % seconds

    % Pick frame to illustrate (max hf_ratio by default)
    if nargin < 3 || isempty(frameIdx)
        [~, frameIdx] = max(hf_ratio);
    end
    frameIdx = max(1, min(frameIdx, nFrames));

    % --- Single-frame spectrum ---
    xk   = double(X(:,frameIdx)) - mean(double(X(:,frameIdx))); % DC remove
    nfft = 2^nextpow2(2*winLen);
    halfN = floor(nfft/2)+1;
    Xf   = fft(xk, nfft);
    P1   = abs(Xf(1:halfN)).^2 / max(1, winLen);                % single-sided power
    f    = (0:halfN-1) * (fs/nfft);                             % Hz
    pdb  = 10*log10(P1 + eps);

    % HF band: >= 4 kHz (or upper half of spectrum if Nyquist < 4 kHz)
    nyq = fs/2;
    if nyq >= 4000
        hf_lo = 4000;
    else
        hf_lo = 0.5 * nyq;  % upper half if sampling rate is low
    end
    if hf_lo >= nyq, hf_lo = 0.9*nyq; end
    hfMask = f >= hf_lo;

    % --- Figure ---
    figure('Color','w');
    tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

    % (1) Spectrum with HF band shaded
    ax1 = nexttile; hold(ax1,'on'); grid(ax1,'on');
    yl = [min(pdb)-3, max(pdb)+3];
    ylim(ax1, yl);
    % shade HF band
    patch(ax1, [hf_lo nyq nyq hf_lo], [yl(1) yl(1) yl(2) yl(2)], ...
          [1 0 0], 'FaceAlpha', 0.08, 'EdgeColor','none');
    % plot spectrum on top
    plot(ax1, f, pdb, 'k-', 'LineWidth', 1.2);
    xlim(ax1, [0 nyq]);
    ylabel(ax1, 'Power (dB)');
    title(ax1, sprintf('Frame %d spectrum — HF band \\geq %.0f Hz | HF/Total = %.2f', ...
                       frameIdx, hf_lo, hf_ratio(frameIdx)));

    % (2) hf_ratio over time
    ax2 = nexttile; hold(ax2,'on'); grid(ax2,'on');
    plot(ax2, t, hf_ratio, 'k-', 'LineWidth', 1.5);
    ylim(ax2, [0 1]); xlim(ax2, [t(1) t(end)]);
    xlabel(ax2, 'Time (s)'); ylabel(ax2, 'HF ratio (0–1)');
    title(ax2, 'High-frequency energy ratio over time');

end

function plot_melPC1(processedAudio, audioBlock)
% Visualise high-band mel features and PC1.
% (A) High-band triangular mel filterbank (≥4 kHz to ~Nyquist).
% (B) Heatmap of high-band log-mel energies (bands × time) with PC1(t).
%
% Inputs:
%   processedAudio.frameSnippets : [winLen x nFrames]
%   processedAudio.sampleRate    : scalar
%   audioBlock(5,:)              : melPC1 scores (optional overlay)

    % ---- Inputs ----
    X  = processedAudio.frameSnippets;   % [winLen x nFrames]
    fs = processedAudio.sampleRate;
    [winLen, nFrames] = size(X);
    t = (0:nFrames-1) * (winLen/fs);     % seconds

    % ---- Rebuild high-band mel filterbank (matches extractor) ----
    nfft   = 2^nextpow2(2*winLen);
    half_n = floor(nfft/2) + 1;
    freqs  = (0:half_n-1) * (fs/nfft);
    nyq    = fs/2;

    % High-band range (≥4 kHz; fallback if fs is low)
    f_lo = max(4000, freqs(2));
    f_hi = 0.98*nyq;
    if f_hi <= f_lo
        f_lo = 0.75*nyq;
        f_hi = 0.98*nyq;
    end

    % 20 triangular mel filters in [f_lo, f_hi]
    mel    = @(f) 2595*log10(1 + f/700);
    invmel = @(m) 700*(10.^(m/2595) - 1);
    nMel   = 20;
    m_edges = linspace(mel(f_lo), mel(f_hi), nMel+2);
    f_edges = invmel(m_edges);

    M = zeros(nMel, half_n);
    for b = 1:nMel
        fL = f_edges(b);   fC = f_edges(b+1);  fR = f_edges(b+2);
        L = (freqs >= fL) & (freqs <= fC);
        R = (freqs >= fC) & (freqs <= fR);
        M(b,L) = (freqs(L) - fL) / max(fC - fL, eps);
        M(b,R) = (fR - freqs(R)) / max(fR - fC, eps);
    end

    % ---- Per-frame high-band log-mel energies ----
    mel_logE = zeros(nMel, nFrames);
    for k = 1:nFrames
        xk = double(X(:,k)) - mean(double(X(:,k)));
        Xk = fft(xk, nfft);
        pow = abs(Xk(1:half_n)).^2;      % single-sided power
        melE = M * pow(:);               % [nMel x 1]
        mel_logE(:,k) = log(melE + eps); % natural log
    end

    % ---- PC1 via SVD (frames × bands), with variance explained ----
    Y  = mel_logE.';                     % [frames x nMel]
    Yc = bsxfun(@minus, Y, mean(Y,1));   % center bands
    [U,S,~] = svd(Yc, 'econ');
    pc1     = U(:,1) * S(1,1);           % same construction as extractor
    svals   = diag(S);
    expl1   = (svals(1)^2) / max(sum(svals.^2), eps);

    % Optional overlay from audioBlock (smoothed), if present
    pc1_overlay = [];
    if nargin >= 2 && ~isempty(audioBlock) && size(audioBlock,1) >= 5
        pc1_overlay = audioBlock(5,:);   % melPC1 row from extractor
    end

    % ---- Figure ----
    figure('Color','w');
    tl = tiledlayout(2,1, 'Padding','compact', 'TileSpacing','compact');

    % (A) High-band mel triangles
    ax1 = nexttile(tl,1); hold(ax1,'on'); grid(ax1,'on');
    fk = freqs/1000;  nyqk = nyq/1000;
    for b = 1:nMel
        mask = M(b,:) > 0;
        plot(ax1, fk(mask), M(b,mask), 'k-');
    end
    xline(ax1, f_lo/1000, ':', 'Color',[0.6 0.6 0.6]);
    xline(ax1, f_hi/1000, ':', 'Color',[0.6 0.6 0.6]);
    xlim(ax1, [0 nyqk]);
    ylim(ax1, [0 1.05]);
    xlabel(ax1, 'Frequency (kHz)');
    ylabel(ax1, 'Filter gain');
    title(ax1, sprintf('High-band mel filterbank (%.0f–%.0f Hz, %d bands)', f_lo, f_hi, nMel));

    % (B) Log-mel energies (bands × time) with PC1(t)
    ax2 = nexttile(tl,2);
    imagesc(ax2, t, 1:nMel, mel_logE); axis(ax2,'xy');
    colormap(ax2, parula);
    colorbar(ax2); ylabel(ax2, 'Mel band'); xlabel(ax2, 'Time (s)');
    title(ax2, sprintf('High-band log-mel energies with PC1(t) — PC1 explains %.1f%%', 100*expl1));

    % Overlay PC1 on right y-axis (z-scored for scale)
    ax2b = axes('Position', ax2.Position, 'Color','none', 'YAxisLocation','right', ...
                'XLim', ax2.XLim, 'YLim', [-3 3], 'XTickLabel', []);
    hold(ax2b,'on');
    pc1z = (pc1 - mean(pc1)) / max(std(pc1), eps);
    plot(ax2b, t, pc1z, 'k-', 'LineWidth', 1.5);

    % Optional dashed overlay from audioBlock
    if ~isempty(pc1_overlay)
        pc1z2 = (pc1_overlay(:) - mean(pc1_overlay(:))) / max(std(pc1_overlay(:)), eps);
        plot(ax2b, t, pc1z2, 'k--', 'LineWidth', 1.0);
        legend(ax2b, {'PC1 (recomputed)','PC1 (from audioBlock)'}, 'Location','southoutside');
    else
        legend(ax2b, {'PC1 (z-score)'}, 'Location','southoutside');
    end
    ylabel(ax2b, 'PC1 (z)');

    linkaxes([ax2, ax2b], 'x');
end
