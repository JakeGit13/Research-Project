function cleanAudio = processAudio(audioFilePath)
    % Load the audio file
    [Y, FS] = audioread(audioFilePath);
    
    % Clean by subtracting scanner noise
    % Channel 1 has scanner noise, Channel 2 has voice+noise
    % Subtracting removes the common noise
    cleanAudio = Y(:,2) - Y(:,1);
    
end


