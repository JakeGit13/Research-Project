function alignmentTest()
% alignmentCheck(incrementValue)
% - Uses a fixed manifest (dataIdx + WAV path) you already mapped.
% - incrementValue is 0-based: 0 selects the first manifest entry, 1 the second, etc.
% - Displays audio waveform (with moving cursor), video frame, MRI frame.
% - Single "Play" button to run through the sequence once in real time.

   
    clc;

    incrementValue = 0;
    

    %% Paths and data
    dataDir = '/Users/jaker/Research-Project/data';
    S = load(fullfile(dataDir, 'mrAndVideoData.mat'), 'data');
    data = S.data;

    %% Manifest (ordered; first item is the one you requested)
    manifest = [ ...
        entry( 9, 'C:\Users\jaker\Research-Project\data\Audio\Raw Audio\sub8_sen_256_6_svtimriMANUAL.wav', 'sub8_sen_256_6');  % 1
        entry( 1, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub8_sen_252_18_svtimriMANUAL.wav',  'sub8_sen_252_18');  % 2       This audio needs to be swapped with...
        entry( 5, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub1_sen_252_1_svtimriMANUAL.wav', 'sub1_sen_252_1'); % 3       ...this audio
        entry( 6, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub8_sen_253_18_svtimriMANUAL.wav', 'sub8_sen_253_18'); % 4
        entry( 7, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub8_sen_254_15_svtimriMANUAL.wav', 'sub8_sen_254_15'); % 5
        entry( 8, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub8_sen_255_17_svtimriMANUAL.wav', 'sub8_sen_255_17'); % 6
        entry(10, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub8_sen_257_15_svtimriMANUAL.wav', 'sub8_sen_257_15'); % 7
        entry(11, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub8_sen_258_8_svtimriMANUAL.wav',  'sub8_sen_258_8');  % 8
        entry(12, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub8_sen_259_18_svtimriMANUAL.wav', 'sub8_sen_259_18'); % 9
        entry(13, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub8_sen_260_1_svtimriMANUAL.wav',  'sub8_sen_260_1');  % 10
        entry(14, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub8_sen_261_15_svtimriMANUAL.wav', 'sub8_sen_261_15'); % 11
        entry(18, 'C:\Users\jaker\Research-Project\data\audio\Raw Audio\sub14_sen_252_14_svtimriMANUAL.wav','sub14_sen_252_14') % 12
    ];

    idx = incrementValue + 1;
    if idx < 1 || idx > numel(manifest)
        error('incrementValue out of range. Valid: 0..%d', numel(manifest)-1);
    end
    M = manifest(idx);

    %% Load audio (stereo clean = R-L)
    [Y, fs] = audioread(M.wav);
    if size(Y,2) == 2
        audio = Y(:,2) - Y(:,1);
    else
        audio = Y;
    end
    audio = audio(:);
    audioDur = numel(audio)/fs;
    tAudio = (0:numel(audio)-1)/fs;

    %% Load frames
    ii = M.dataIdx;
    videoFrames = data(ii).video_frames;   % 1xT cells
    mrFrames    = data(ii).mr_frames;      % 1xT cells
    T = numel(videoFrames);

    % Derive fps from durations (avoids hard-coding)
    frameRate = T / audioDur;
    vidDur = (T-1)/frameRate;

    fprintf('Item %d/%d: dataIdx=%d | %s | T=%d | fps=%.3f | audio=%.2fs | video=%.2fs\n', ...
        idx, numel(manifest), ii, M.label, T, frameRate, audioDur, vidDur);

    %% Figure

    % Kill any old alignment windows
    delete(findall(0, 'Type', 'figure', 'Name', 'Alignment Window'));
    




    fig = figure('Position',[100 100 1200 800], 'Name',['Alignment: ' M.label], 'NumberTitle','off');

    % Audio plot
    subplot(3,1,1);
    plot(tAudio, audio); hold on;
    xlabel('Time (s)'); ylabel('Amplitude');
    title(['Audio — ' M.label], 'Interpreter','none');
    xlim([0 max(audioDur, vidDur)]);
    hLine = xline(0,'r','LineWidth',2);

    % Video frame
    subplot(3,1,2);
    hVid = imshow(videoFrames{1});
    title(sprintf('Video Frame 1/%d', T));

    % MRI frame
    subplot(3,1,3);
    hMR  = imshow(mrFrames{1});
    title(sprintf('MRI Frame 1/%d', T));

    % Play button (single control)
    uicontrol('Style','pushbutton','String','Play', ...
              'Position',[20 20 60 30], 'Callback',@(~,~) playOnce());

    %% Nested: play sequence once
    function playOnce()
        ap = audioplayer(audio, fs);
        play(ap);
        t0 = tic;
        for f = 1:T
            t = (f-1)/frameRate;
            if ~ishandle(fig), break; end
            hLine.Value = t;
            set(hVid,'CData',videoFrames{f});
            set(hMR, 'CData',mrFrames{f});
            drawnow;

            % Real-time pacing
            elapsed = toc(t0);
            if t > elapsed, pause(t - elapsed); end
        end
        if isvalid(ap) && isplaying(ap), stop(ap); end
    end
end

%% Helper for manifest rows
function s = entry(dataIdx, wavPath, label)
    s = struct('dataIdx',dataIdx, 'wav',wavPath, 'label',label);
end
