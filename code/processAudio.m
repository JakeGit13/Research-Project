function processedAudio = processAudio(wavPath, nFrames, opts)   
    
    % Default arguments 
    arguments
        wavPath (1,:) char
        nFrames (1,1) double 
        opts.VERBOSE (1,1) logical = false  % false as default
        opts.genFigures (1,1) logical = false  % false as default
        
    end
    
    VERBOSE = opts.VERBOSE;
    genFigures = opts.genFigures;   
 

    %% Load audio & remove background noise (SHOULD PLOT SOMETHING HERE)
    [originalAudio, sampleRate] = audioread(wavPath);
    % Simple scanner-noise cancellation by channel subtraction 
    cleanAudio = originalAudio(:,2) - originalAudio(:,1);
    cleanAudio = cleanAudio(:);
    cleanAudio = cleanAudio - mean(cleanAudio);   % remove DC offset 



    %% Preprocessing ===================================
    
    % Basic info
    nSamples         = numel(cleanAudio);
    audioDurationSec = nSamples / sampleRate;
    
    if VERBOSE
        fprintf('Audio duration     : %.3f s\n', audioDurationSec);
        fprintf('nFrames (MR)               : %d\n', nFrames);
    end
    
    %% Fixed 50 ms frame-centred snippets (even split across audio)
    
    % Even split in SAMPLES and centre locations (half-step offset)
    samplesPerFrame = nSamples / nFrames;                          % samples per MR frame
    centerSampleIdx = (0:nFrames-1) * samplesPerFrame + samplesPerFrame/2;
    centerSampleIdx = round(centerSampleIdx) + 1;                  % 1-based indexing
    centerSampleIdx = max(1, min(centerSampleIdx, nSamples));      % clamp to [1, nSamples]
    
    % Fixed 50 ms Hann window (force odd length so we have a true centre sample)
    windowLen  = round(0.050 * sampleRate);
    if mod(windowLen,2)==0, windowLen = windowLen + 1; end
    halfWindow = (windowLen - 1)/2;
    hannWindow = hann(windowLen, 'symmetric');
    
    if VERBOSE
        fprintf('Window length              : %d samples (%.1f ms)\n', ...
            windowLen, 1000*windowLen/sampleRate);
    end
    
    % Zero-pad to handle edge frames 
    paddedAudio     = [zeros(halfWindow,1); cleanAudio; zeros(halfWindow,1)];
    centerIdxPadded = centerSampleIdx + halfWindow;
    
    % Extract one windowed snippet per MR frame (columns = frames)
    frameSnippets = zeros(windowLen, nFrames, 'like', cleanAudio);
    for k = 1:nFrames
        iLo = centerIdxPadded(k) - halfWindow;
        iHi = centerIdxPadded(k) + halfWindow;
        frameSnippets(:,k) = paddedAudio(iLo:iHi) .* hannWindow;
    end
    
    fprintf('Extracted %d snippets of %d samples each (~%.1f ms)\n', ...
        nFrames, windowLen, 1000*windowLen/sampleRate);



    % ---- Return struct for feature extraction ---- 
    processedAudio.sampleRate          = sampleRate;
    processedAudio.windowLengthSamples = windowLen;
    processedAudio.centerSampleIdx     = centerSampleIdx(:);   % nFrames×1
    processedAudio.frameSnippets       = frameSnippets;        % winLen×nFrames
    processedAudio.nFrames             = nFrames;
    processedAudio.durationSec         = audioDurationSec;
    processedAudio.windowMs      = 1000 * processedAudio.windowLengthSamples / processedAudio.sampleRate;
    processedAudio.centerTimeSec = (processedAudio.centerSampleIdx - 1) / processedAudio.sampleRate;



    %% Optional visualisation (pre vs post), controlled by opts.PLOTTING
    if genFigures
        % --- choose a "raw" channel for display (robust if mono) ---
        if size(originalAudio,2) >= 2
            rawDisp = originalAudio(:,2);    % typical "speech+scanner" channel
        else
            rawDisp = originalAudio(:,1);
        end
        rawDisp   = rawDisp(:) - mean(rawDisp(:));
        cleanDisp = cleanAudio(:);           % already DC-removed above

        t_raw   = (0:numel(rawDisp)-1)/sampleRate;
        t_clean = (0:numel(cleanDisp)-1)/sampleRate;

        % frame centres in seconds (shared across panels)
        t_centres = (centerSampleIdx - 1)/sampleRate;

        % pick three illustrative frames: first, middle, last
        idx_show = unique(round([1, nFrames/2, nFrames]));
        idx_show = min(max(idx_show,1), nFrames);
        halfW    = halfWindow;   % samples

        figure('Color','w');
        tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

        % ---------- (1) RAW waveform ----------
        ax1 = nexttile(tl,1); hold(ax1,'on'); grid(ax1,'on');
        plot(ax1, t_raw, rawDisp, 'Color',[0.2 0.2 0.2]);
        % frame-centre markers
        for k = 1:numel(t_centres), xline(ax1, t_centres(k), ':', 'Color',[0.8 0.8 0.8]); end
        % shade 50 ms windows for three example frames
        % --- in RAW waveform panel (ax1) ---
        for k = idx_show
            c   = t_centres(k);
            xlo = c - 0.5*windowLen/sampleRate;
            xhi = c + 0.5*windowLen/sampleRate;
            yL  = ylim(ax1);                            % [ymin ymax]
            patch(ax1, [xlo xhi xhi xlo], [yL(1) yL(1) yL(2) yL(2)], ...
                  [0 0.5 1], 'FaceAlpha', 0.08, 'EdgeColor', 'none');
        end

        xlabel(ax1,'Time (s)'); ylabel(ax1,'Amplitude');
        title(ax1, sprintf('Raw waveform (fs=%d Hz) — %d frames, window %.0f ms', ...
            sampleRate, nFrames, 1000*windowLen/sampleRate));

        % ---------- (2) CLEAN waveform ----------
        ax2 = nexttile(tl,2); hold(ax2,'on'); grid(ax2,'on');
        plot(ax2, t_clean, cleanDisp, 'k');
        for k = 1:numel(t_centres), xline(ax2, t_centres(k), ':', 'Color',[0.8 0.8 0.8]); end
        % --- in CLEAN waveform panel (ax2) ---
        for k = idx_show
            c   = t_centres(k);
            xlo = c - 0.5*windowLen/sampleRate;
            xhi = c + 0.5*windowLen/sampleRate;
            yL  = ylim(ax2);                            % [ymin ymax]
            patch(ax2, [xlo xhi xhi xlo], [yL(1) yL(1) yL(2) yL(2)], ...
                  [0 1 0], 'FaceAlpha', 0.08, 'EdgeColor', 'none');
        end

        xlabel(ax2,'Time (s)'); ylabel(ax2,'Amplitude');
        title(ax2, 'Cleaned waveform (channel subtraction + DC removal)');

        % ---------- helper: spectrogram with consistent settings ----------
        winSG   = hann(windowLen,'symmetric');
        ovSG    = round(0.75*windowLen);
        nfftSG  = windowLen;  % keep it simple/fast

        % compute raw spectrogram
        [Sraw,Fraw,Traw]   = spectrogram(rawDisp,   winSG, ovSG, nfftSG, sampleRate, 'yaxis');
        SrawDB = 20*log10(abs(Sraw)+eps);
        % compute clean spectrogram
        [Scln,Fcln,Tcln]   = spectrogram(cleanDisp, winSG, ovSG, nfftSG, sampleRate, 'yaxis');
        SclnDB = 20*log10(abs(Scln)+eps);

        % unify colour limits for fair comparison (95th percentile of both)
        hi = prctile([SrawDB(:); SclnDB(:)], 95);
        lo = hi - 60;

        % ---------- (3) RAW spectrogram ----------
        ax3 = nexttile(tl,3);
        imagesc(ax3, Traw, Fraw, SrawDB); axis(ax3,'xy');
        caxis(ax3,[lo hi]); colormap(ax3, parula);
        xlabel(ax3,'Time (s)'); ylabel(ax3,'Frequency (Hz)');
        title(ax3,'Raw spectrogram');

        % ---------- (4) CLEAN spectrogram ----------
        ax4 = nexttile(tl,4);
        imagesc(ax4, Tcln, Fcln, SclnDB); axis(ax4,'xy');
        caxis(ax4,[lo hi]); colormap(ax4, parula);
        xlabel(ax4,'Time (s)'); ylabel(ax4,'Frequency (Hz)');
        title(ax4,'Cleaned spectrogram');

        linkaxes([ax1,ax2],'x');  % keep time alignment
    end


end






