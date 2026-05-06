function repeat_generate(channelType, snrList, overwrite)
% Repeat generate_wifi_lltf_dataset over SNR values for one channel type.
%
% Output files:
%   dataset_<channelType>_<snr>db.csv
%   dataset_<channelType>_<snr>db_eval.csv

if nargin < 1 || isempty(channelType)
    channelType = 'rayleigh';
end

if nargin < 2 || isempty(snrList)
    snrList = 0:3:18;
end

if nargin < 3 || isempty(overwrite)
    overwrite = false;
end

channelType = lower(string(channelType));

assert(isscalar(channelType), ...
    'channelType must be one scalar string or char.');

assert(ismember(channelType, ["onetap", "rayleigh", "rician"]), ...
    'channelType must be onetap, rayleigh, or rician.');

channelTypeChar = char(channelType);
snrList = snrList(:);

numJobs = numel(snrList);

for i = 1:numJobs
    snrDb = snrList(i);
    snrRound = round(snrDb);

    rawMainFile = sprintf('dataset_%ddb.csv', snrRound);
    rawEvalFile = sprintf('dataset_%ddb_eval.csv', snrRound);

    typedMainFile = sprintf('dataset_%s_%ddb.csv', ...
        channelTypeChar, snrRound);
    typedEvalFile = sprintf('dataset_%s_%ddb_eval.csv', ...
        channelTypeChar, snrRound);

    fprintf('\n[%d/%d] channelType = %s, SNR = %d dB\n', ...
        i, numJobs, channelTypeChar, snrRound);

    if isfile(typedMainFile) && isfile(typedEvalFile) && ~overwrite
        fprintf('  skipped: files already exist\n');
    elseif (isfile(typedMainFile) || isfile(typedEvalFile)) && ~overwrite
        error('Partial output files exist. Use overwrite=true or delete them manually.');

    else
        if isfile(rawMainFile)
            delete(rawMainFile);
        end

        if isfile(rawEvalFile)
            delete(rawEvalFile);
        end

        if overwrite
            if isfile(typedMainFile)
                delete(typedMainFile);
            end

            if isfile(typedEvalFile)
                delete(typedEvalFile);
            end
        end

        generate_wifi_lltf_dataset(snrDb, channelTypeChar);

        if ~isfile(rawMainFile) || ~isfile(rawEvalFile)
            error('Generated files were not found for SNR = %d dB.', snrRound);
        end

        movefile(rawMainFile, typedMainFile, 'f');
        movefile(rawEvalFile, typedEvalFile, 'f');
    end
end
end
