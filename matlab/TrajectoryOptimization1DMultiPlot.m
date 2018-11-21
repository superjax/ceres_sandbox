format compact
set(0,'DefaultFigureWindowStyle','docked')

state_file = fopen('../build/1d_trajectory_multi_states.bin', 'r');
init_file = fopen('../build/1d_trajectory_multi_states_init.bin', 'r');
input_file = fopen('../build/1d_trajectory_multi_input.bin', 'r');

x = reshape(fread(state_file, 'double'), 2, []);
x0 = reshape(fread(init_file, 'double'), 2, []);
input = fread(input_file, 'double');
time = linspace(0, 1, length(x));

%% Plot Trajectory
figure(2); clf;
set(gcf, 'name', 'Trajectory1D', 'NumberTitle', 'off');
subplot(3,1,1); hold on;
ylabel("x (m)")
plot(time, x(1,:))
plot(time, x0(1,:))
legend("x", "x0");
subplot(3,1,2); hold on;
ylabel("v (m/s)")
plot(time, x(2,:))
plot(time, x0(2,:))
subplot(3,1,3); hold on;
ylabel("F (N)");
plot(time, input)
