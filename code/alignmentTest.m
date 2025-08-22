function alignmentTest()
    clc; clear all;


    % Load your data
    dataDir = '/Users/jaker/Research-Project/data';
    load(fullfile(dataDir, 'mrAndVideoData.mat'), 'data');
    audioFile = 'C:\Users\jaker\Research-Project\data\audio\sub8_sen_256_6_svtimriMANUAL.wav';
    
    % Load and clean audio (modified section)
    [Y, fs] = audioread(audioFile);
    
    % Check if audio is stereo and clean it
    if size(Y, 2) == 2
        audio = Y(:,2) - Y(:,1);  % Clean audio by subtracting channels
        fprintf('Audio cleaned: subtracted channel 1 from channel 2\n');
    else
        audio = Y;  % If mono, use as is
        fprintf('Audio is mono, using as is\n');
    end
    
    % Ensure audio is a column vector for consistency
    audio = audio(:);
    
    audioTime = (0:length(audio)-1) / fs;



    
    % Get video/MRI data
    dataIdx = 9;
    videoFrames = data(dataIdx).video_frames;  % 1x40 cell array
    mrFrames = data(dataIdx).mr_frames;        % 1x40 cell array
    nFrames = length(videoFrames);  % 40 frames
    
    % Frame rate calculation - with 40 frames, we need to estimate based on audio duration
    % Or you can set this if you know it
    frameRate = 15.5; % Adjust if you know the actual rate
    % Alternative: calculate based on audio duration
    % frameRate = nFrames / audioTime(end);
    
    videoTime = (0:nFrames-1) / frameRate;
    
    % Print some info
    fprintf('Audio duration: %.2f seconds\n', audioTime(end));
    fprintf('Video duration (at %d fps): %.2f seconds\n', frameRate, videoTime(end));
    fprintf('Number of frames: %d\n', nFrames);
    
    % Create synchronized visualization
    fig = figure('Position', [100 100 1200 800]);
    
    % Audio waveform
    subplot(3,1,1);
    plot(audioTime, audio);
    hold on;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Audio Waveform');
    audioLine = xline(0, 'r', 'LineWidth', 2);
    
    % Video frame
    subplot(3,1,2);
    vidImg = imshow(videoFrames{1});
    title('Video Frame 1/40');
    
    % MRI frame
    subplot(3,1,3);
    mrImg = imshow(mrFrames{1});
    title('MRI Frame 1/40');
    
    % Add slider for time navigation
    slider = uicontrol('Style', 'slider', 'Position', [100 20 1000 30]);
    slider.Min = 0;
    slider.Max = min(audioTime(end), videoTime(end));
    slider.Value = 0;
    
    % Add text to show current time and frame
    timeText = uicontrol('Style', 'text', 'Position', [550 50 200 20]);
    timeText.String = sprintf('Time: 0.00 s | Frame: 1/%d', nFrames);
    
    % Store handles in guidata for access in nested functions
    handles = struct();
    handles.slider = slider;
    handles.audioLine = audioLine;
    handles.vidImg = vidImg;
    handles.mrImg = mrImg;
    handles.timeText = timeText;
    handles.videoFrames = videoFrames;
    handles.mrFrames = mrFrames;
    handles.nFrames = nFrames;
    handles.frameRate = frameRate;
    handles.audio = audio;
    handles.fs = fs;
    handles.audioTime = audioTime;
    guidata(fig, handles);
    
    % Update function
    slider.Callback = @updateVisualization;
    
    % Add play button
    playButton = uicontrol('Style', 'pushbutton', 'String', 'Play', ...
        'Position', [20 20 60 30], 'Callback', @playSequence);
    
    % Add frame step buttons
    prevButton = uicontrol('Style', 'pushbutton', 'String', '< Prev', ...
        'Position', [90 60 60 30], 'Callback', @prevFrame);
    nextButton = uicontrol('Style', 'pushbutton', 'String', 'Next >', ...
        'Position', [160 60 60 30], 'Callback', @nextFrame);
    
    % Add audio play button for current position
    playAudioButton = uicontrol('Style', 'pushbutton', 'String', 'Play Audio Segment', ...
        'Position', [230 60 120 30], 'Callback', @playAudioSegment);
end

function updateVisualization(src, ~)
    handles = guidata(src);
    currentTime = src.Value;
    
    % Update audio cursor
    handles.audioLine.Value = currentTime;
    
    % Calculate frame index
    frameIdx = round(currentTime * handles.frameRate) + 1;
    frameIdx = min(max(frameIdx, 1), handles.nFrames);
    
    % Update time text
    handles.timeText.String = sprintf('Time: %.2f s | Frame: %d/%d', ...
        currentTime, frameIdx, handles.nFrames);
    
    % Update video frame
    handles.vidImg.CData = handles.videoFrames{frameIdx};
    subplot(3,1,2);
    title(sprintf('Video Frame %d/%d', frameIdx, handles.nFrames));
    
    % Update MRI frame
    handles.mrImg.CData = handles.mrFrames{frameIdx};
    subplot(3,1,3);
    title(sprintf('MRI Frame %d/%d', frameIdx, handles.nFrames));
    
    drawnow;
end

function playSequence(src, ~)
    handles = guidata(src);
    startTime = handles.slider.Value;
    endTime = handles.slider.Max;
    timeStep = 1/handles.frameRate;
    
    % Option to play with audio
    playWithAudio = 'Yes';
    
    if strcmp(playWithAudio, 'Yes')
        % Calculate starting audio sample
        startSample = round(startTime * handles.fs) + 1;
        startSample = min(max(startSample, 1), length(handles.audio));
        
        % Play audio from current position
        audioPlayer = audioplayer(handles.audio(startSample:end), handles.fs);
        play(audioPlayer);
    end
    
    tic;
    for t = startTime:timeStep:endTime
        handles.slider.Value = t;
        updateVisualization(handles.slider, []);
        
        % Try to maintain real-time playback
        elapsed = toc;
        targetTime = t - startTime;
        if targetTime > elapsed
            pause(targetTime - elapsed);
        end
        
        % Check if figure was closed
        if ~ishandle(handles.slider)
            if strcmp(playWithAudio, 'Yes') && isplaying(audioPlayer)
                stop(audioPlayer);
            end
            break;
        end
    end
    
    if strcmp(playWithAudio, 'Yes') && exist('audioPlayer', 'var') && isplaying(audioPlayer)
        stop(audioPlayer);
    end
end

function prevFrame(src, ~)
    handles = guidata(src);
    currentTime = handles.slider.Value;
    newTime = max(0, currentTime - 1/handles.frameRate);
    handles.slider.Value = newTime;
    updateVisualization(handles.slider, []);
end

function nextFrame(src, ~)
    handles = guidata(src);
    currentTime = handles.slider.Value;
    newTime = min(handles.slider.Max, currentTime + 1/handles.frameRate);
    handles.slider.Value = newTime;
    updateVisualization(handles.slider, []);
end

function playAudioSegment()   % Change this to just make it go back to the start somehow 
    handles.slider.Value = 0.0;
    updateVisualization(handles.slider, []);
end