format compact
set(0,'DefaultFigureWindowStyle','docked')

truth_file = fopen('../build/Imu3d.MultiWindow.truth.log', 'r');
est_file = fopen('../build/Imu3d.MultiWindow.est.log', 'r');

truth = fread(truth_file, 'double');
est = fread(est_file, 'double');
truth = reshape(truth, 11, []);
est = reshape(est, 11, []);

names = ["t", "px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz"];


%% Plot position
figure(1); clf;
set(gcf, 'name', 'Position', 'NumberTitle', 'off');
for i = 1:3
    idx = i+1;
    subplot(3,1,i);
    plot(truth(1,:), truth(idx,:));
    hold on;
    plot(est(1,:), est(idx,:));
    title(names(idx));
    legend("truth", "est")
end

%% Plot attitude
figure(2); clf;
set(gcf, 'name', 'Attitude', 'NumberTitle', 'off');
for i = 1:4
    idx = i+4;
    subplot(4,1,i);
    plot(truth(1,:), truth(idx,:));
    hold on;
    plot(est(1,:), est(idx,:));
    title(names(idx));
    legend("truth", "est")
end


%% Plot velocity
figure(3); clf;
set(gcf, 'name', 'Velocity', 'NumberTitle', 'off');
for i = 1:3
    idx = i+8;
    subplot(3,1,i);
    plot(truth(1,:), truth(idx,:));
    hold on;
    plot(est(1,:), est(idx,:));
    title(names(idx));
    legend("truth", "est")
end