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

        plot_F0_voicing(processedAudio, audioBlock) 

        
    end




    %% Optional: RETURN noise block instead of real features
    % useNoiseAudio = isfield(processedAudio,'useNoiseAudio') && logical(processedAudio.useNoiseAudio);
    if useNoiseAudio
        rng(12345);  % reproducible control
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

    % ---- Inputs expected ----
    % processedAudio.frameSnippets : [winLen x nFrames]
    % processedAudio.sampleRate    : scalar
    % processedAudio.vuvThresh     : (optional) scalar, default 0.4
    % audioBlock(1,:) = logF0;  audioBlock(2,:) = vuvProb

    X  = processedAudio.frameSnippets;
    fs = processedAudio.sampleRate;
    if isfield(processedAudio,'vuvThresh')
        vuvThresh = processedAudio.vuvThresh;
    else
        vuvThresh = 0.40;
    end

    [winLen, nFrames] = size(X);
    t = (0:nFrames-1) * (winLen/fs);                 % seconds

    % ---- Per-frame spectrogram (no raw wav needed) ----
    nfft   = 2^nextpow2(2*winLen);
    halfN  = floor(nfft/2)+1;
    specDB = zeros(halfN, nFrames);
    for k = 1:nFrames
        xk = X(:,k) - mean(X(:,k));                  % DC remove
        mag = abs(fft(xk, nfft));
        specDB(:,k) = 20*log10(mag(1:halfN) + eps);
    end
    f = (0:halfN-1) * (fs/nfft);                     % Hz
    climHi = prctile(specDB(:),95);                  % display range
    climLo = climHi - 60;

    % ---- Features ----
    logF0   = audioBlock(1,:);
    vuvProb = audioBlock(2,:);
    f0_hz = exp(logF0);
    f0_hz(~isfinite(f0_hz) | f0_hz<=0) = NaN;        % hide invalid
    voiced = (vuvProb >= vuvThresh);

    % ---- Plot ----
    figure('Color','w');

    % (A) Spectrogram with F0 overlay
    subplot(2,1,1);
    imagesc(t, f, specDB); axis xy;
    colormap parula; caxis([climLo climHi]);
    hold on;
    f0_voiced = f0_hz;  f0_voiced(~voiced) = NaN;
    f0_unv    = f0_hz;  f0_unv(voiced)    = NaN;
    plot(t, f0_voiced, 'k',  'LineWidth', 1.5);
    plot(t, f0_unv,   'k--','LineWidth', 1.0);
    ylim([0, min(5000, fs/2)]);
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title('Spectrogram with F0 (solid=voiced, dashed=interpolated)');

    % (B) Voicing probability strip
    subplot(2,1,2);
    plot(t, vuvProb, 'k','LineWidth',1.2); hold on;
    yline(vuvThresh, '--r');
    ylim([0 1]); xlim([t(1) t(end)]);
    xlabel('Time (s)'); ylabel('Voicing prob.');
    title(sprintf('Voicing (threshold = %.2f)', vuvThresh));

end
