% ADD INFO HERE 

clearvars; clc;

%% Paths =====
projectRoot = '/Users/jaker/Research-Project/data';     % This should be the only thing that the user needs to set up? e.g. path to research project?  
addpath(projectRoot);

% Load in MR and video data struct
results = load(fullfile(projectRoot, 'mrAndVideoData.mat'), 'data');   % MR / video data struct 
mrAndVideoData = results.data;

% Path to raw audio folder
audioFolderPath = fullfile(projectRoot, 'Audio', 'Raw Audio');     % Where raw audio files are located

resultsRoot = fullfile(projectRoot, 'results'); % Where H1 / H2 results save to 
h1CSV = fullfile(resultsRoot, 'h1_results.csv');
h2CSV = fullfile(resultsRoot, 'h2_results.csv');


%% Controls ====
nBoots = 10;    % Universal across all tests (1000 as default)

generateCsv = true;

doH1 = false;
doH2 = false;

% Independent switches to write CSVs 
writeToCsv_h1 = false;
writeToCsv_h2 = false;


% Corrected map between MR/Video and Audio files [dataIdx, filename]
manifest = {
     9,  'sub8_sen_256_6_svtimriMANUAL.wav';
     1,  'sub8_sen_252_18_svtimriMANUAL.wav';   % Swapped dataIdx with ...
     5,  'sub1_sen_252_1_svtimriMANUAL.wav';    % ... this dataIdx
     6,  'sub8_sen_253_18_svtimriMANUAL.wav';
     7,  'sub8_sen_254_15_svtimriMANUAL.wav';
     8,  'sub8_sen_255_17_svtimriMANUAL.wav';
    10,  'sub8_sen_257_15_svtimriMANUAL.wav';
    11,  'sub8_sen_258_8_svtimriMANUAL.wav';
    12,  'sub8_sen_259_18_svtimriMANUAL.wav';
    13,  'sub8_sen_260_1_svtimriMANUAL.wav';
    14,  'sub8_sen_261_15_svtimriMANUAL.wav';
    18,  'sub14_sen_252_14_svtimriMANUAL.wav'
};


manifestLength = size(manifest,1) / 12 ; % 3 sentences for testing purposes 


for i = 1:manifestLength        % Loop through all 12 sentences using manifest

    fprintf("Run %d/%d\n",i,manifestLength);

    % Get necessary attributes for sentence i of the manifest 
    dataIdx = manifest{i,1};
    wavName = manifest{i,2};

    % Get direct path to .wav file of sentence i by concatenating 
    wavPath = fullfile(audioFolderPath, wavName);

    nFrames = size(mrAndVideoData(dataIdx).mr_warp2D, 2); % Number of MR / video frames 

    actorID    = mrAndVideoData(dataIdx).actor;   
    sentenceID = mrAndVideoData(dataIdx).sentence;

    % Returns audio features for this sentence
    audioFeatures = extractAudioFeatures(wavPath,nFrames);

    
    if doH1
        fprintf("Starting H1\n");
    
        % observedMode + shuffleTarget pairs, add more test conditions here
        h1TestsParameters = {
            "MR+VID", 1;
            "MR",     1;
            "VID",    2
        };
    
        for k = 1:size(h1TestsParameters,1)
            observedMode = h1TestsParameters{k,1};
            shuffleTarget = h1TestsParameters{k,2};
    
            r = trimodalH1(mrAndVideoData, audioFeatures, dataIdx, ...
                           reconstructId=3, shuffleTarget=shuffleTarget, ...
                           observedMode=observedMode, nBoots=nBoots);
    
            if ~isfile(h1CSV), generateEmptyCSV(r, h1CSV); end
            if writeToCsv_h1 && isfile(h1CSV), appendToCSV(r, h1CSV); end
        end
    end

    
    
    
    if doH2
        fprintf("Starting H2\n");
    
        r2_mr = trimodalH2(mrAndVideoData, audioFeatures, dataIdx, ...
                           reconstructId=1, shuffleTarget=3, nBoots=nBoots);
        
        if ~isfile(h2CSV), generateEmptyCSV(r2_mr, h2CSV); end % Only generate if no CSV file is there
        if writeToCsv_h2 && isfile(h2CSV); appendToCSV(r2_mr,h2CSV); end   % Append these results for this sentence to the CSV 
    
        r2_vid = trimodalH2(mrAndVideoData, audioFeatures, dataIdx, ...
                            reconstructId=2, shuffleTarget=3, nBoots=nBoots);

        if writeToCsv_h2 && isfile(h2CSV); appendToCSV(r2_vid,h2CSV); end   % Append these results for this sentence to the CSV 
        
    end


    fprintf("Done\n");

end


function appendToCSV(resultStruct, csvPath)

    % Read header (column order)
    options = detectImportOptions(csvPath);
    headerVariables = options.VariableNames;   % cell array of variable names

    % Build one row in header order; flatten non-scalars
    row = cell(1, numel(headerVariables));
    for k = 1:numel(headerVariables)
        name = headerVariables{k};
        v = resultStruct.(name);   % will error naturally if field missing

        if (isnumeric(v) || islogical(v)) && isscalar(v)
            row{k} = v;
        elseif ischar(v) || (isstring(v) && isscalar(v))
            row{k} = v;
        else
            row{k} = string(jsonencode(v));  % flatten arrays/structs/cells
        end
    end

    % Append the row
    T = cell2table(row, 'VariableNames', headerVariables);
    writetable(T, csvPath, 'WriteMode', 'append');
end







function generateEmptyCSV(resultStruct, csvPathIn)
    % Get field Names of the results from trimodal H1 / H2
    fn = fieldnames(resultStruct);

    %  Create a 0-row table with those columns to create the header
    T = cell2table(cell(0, numel(fn)), 'VariableNames', fn);

    % Write header-only CSV (will overwrite if exists)
    writetable(T, csvPathIn);

    fprintf('Created empty CSV at %s with %d columns.\n', csvPathIn, numel(fn));
end







