function resultsTbl = run_wifi_lltf_snr_sweep_wrapper()
% Call external functions:
%   1) generate_wifi_lltf_dataset(snrDb, channelType)
%   2) [lsnmae, lmmsenmae] = evaluate_wifi_lltf_ls_mmse_nmae(evalCsv)
%
% SNR sweep: 0, 3, 6, 9, 12, 15, 18 dB

snrList = 0:3:18;

lsnmaeList = zeros(numel(snrList), 1);
lmmsenmaeList = zeros(numel(snrList), 1);

channelType = 'onetap';   % 'onetap', 'rayleigh', 'rician'

for ii = 1:numel(snrList)
    snrDb = snrList(ii);

    fprintf('============================================================\n');
    fprintf('SNR = %d dB\n', snrDb);
    fprintf('channelType = %s\n', channelType);

    generate_wifi_lltf_dataset(snrDb, channelType);

    evalCsv = sprintf('wifi_lltf_dataset_%ddb_eval.csv', round(snrDb));

    [lsnmae, lmmsenmae] = evaluate_wifi_lltf_ls_mmse_nmae(evalCsv);

    lsnmaeList(ii) = lsnmae;
    lmmsenmaeList(ii) = lmmsenmae;
end

resultsTbl = table(snrList(:), lsnmaeList, lmmsenmaeList, ...
    'VariableNames', {'SNR_dB', 'LS_NMAE', 'LMMSE_NMAE'});

csvFile = sprintf('results_%s.csv', channelType);
writetable(resultsTbl, csvFile);
fprintf('Saved result table: %s\n', csvFile);

end
