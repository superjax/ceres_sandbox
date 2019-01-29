format compact
set(0,'DefaultFigureWindowStyle','docked')

file = fopen('/tmp/ceres_sandbox/Imu3D.CheckPropagation.log', 'r');

data = fread(file, 'double');
data = reshape(data, 1+7+3+7+3+6 + 4, [])';

t = data(:, 1);
xhat = data(:, 2:11);
x = data(:, 12:21);
u = data(:, 22:27);
test = data(:, 28:end);

names = ["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz"];


figure(1); clf;
set(gcf, 'name', 'Position', 'NumberTitle', 'off');
for i =1:3
    idx = i;
    subplot(3, 1, i)
    plot(t, x(:,idx))
    hold on;
    plot(t, xhat(:,idx))
    title(names(idx));
    legend("x", "xhat")
end


figure(2); clf;
set(gcf, 'name', 'Attitude', 'NumberTitle', 'off');
for i =1:4
    idx = i+3;
    subplot(4, 1, i)
    plot(t, x(:,idx))
    hold on;
    plot(t, xhat(:,idx))
    plot(t, test(:,i))
    title(names(idx));
    legend("x", "xhat")
end


figure(3); clf;
set(gcf, 'name', 'Velocity', 'NumberTitle', 'off');
for i =1:3
    idx = i+7;
    subplot(3, 1, i)
    plot(t, x(:,idx))
    hold on;
    plot(t, xhat(:,idx))
    title(names(idx));
    legend("x", "xhat")
end

names = ["ax", "ay", "az", "wx", "wy", "wz"];
figure(4); clf;
set(gcf, 'name', 'Input', 'NumberTitle', 'off');
for j = 1:2
    for i =1:3
        idx = i + (j-1)*3;
        subplot(3, 2, (i-1)*2+j)
        plot(t, u(:,idx))
        hold on;
        title(names(idx));
    end
end
