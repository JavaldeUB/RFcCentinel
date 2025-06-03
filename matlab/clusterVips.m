function clusters = clusterVips(vipfeat)
% Initialize clusters
clusters = {};
current_cluster = vipfeat(1); % Start with the first element
% Loop through the vector
for i = 2:length(vipfeat)
    if vipfeat(i) - vipfeat(i-1) == 1
        % Consecutive element, add to current cluster
        current_cluster = [current_cluster, vipfeat(i)];
    else
        % Non-consecutive, save the current cluster and start a new one
        clusters{end+1} = current_cluster; %#ok<*AGROW>
        current_cluster = vipfeat(i);
    end
end

% Save the last cluster
clusters{end+1} = current_cluster;
% Display the clusters
% disp('Clusters:');
% for i = 1:length(clusters)
%     fprintf('VIPs Cluster %d: %s\n', i, mat2str(clusters{i}));
% end
end