format compact
set(0,'DefaultFigureWindowStyle','docked')

state_file = fopen('../build/3d_trajectory_states_init.bin', 'r');
init_file = fopen('../build/3d_trajectory_states.bin', 'r');
input_file = fopen('../build/3d_trajectory_input.bin', 'r');

x = reshape(fread(state_file, 'double'), 10, []);
x0 = reshape(fread(init_file, 'double'), 10, []);
input = reshape(fread(input_file, 'double'), 4, []);
[rows, cols] = size(x0);
time = linspace(0, 1, cols);

labels =["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz", "F"];

%% Plot Position
figure(1); clf;
set(gcf, 'name', 'Trajectory3DPosition', 'NumberTitle', 'off');
for i = 1:3
    subplot(3,1,i); hold on;
    ylabel(labels(i))
    plot(time, x(i,:))
    plot(time, x0(i,:))
end

%% Plot Attitude
figure(2); clf;
set(gcf, 'name', 'Trajectory3DAttitude', 'NumberTitle', 'off');
for i = 1:4
    idx = i + 3;
    subplot(4,1,i); hold on;
    ylabel(labels(idx))
    plot(time, x(idx,:))
    plot(time, x0(idx,:))
end


%% Plot Velocity
figure(3); clf;
set(gcf, 'name', 'Trajectory3DVelocity', 'NumberTitle', 'off');
for i = 1:3
    idx = i + 7;
    subplot(3,1,i); hold on;
    ylabel(labels(idx))
    plot(time, x(idx,:))
    plot(time, x0(idx,:))
end

%% Plot Inputs
figure(4); clf;
set(gcf, 'name', 'Trajectory3DInputs', 'NumberTitle', 'off');
for i = 1:4
    idx = i + 10;
    subplot(4,1,i); hold on;
    ylabel(labels(idx))
    plot(time, input(i,:))
end