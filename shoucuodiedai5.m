% 固定参数设置
M = 1000;
current_ye1 = 0.4;
current_ye2 = 0.6;
current_yab = 0.9;
current_yba = 1.0;

% 辅助函数定义
V = @(y) 1 - (1 + y).^(-2);
C = @(y) log2(1 + y);

% 计算当前信道参数
Ve1 = V(current_ye1);
Ce1 = C(current_ye1);
Vab = V(current_yab);
Cab = C(current_yab);
Ve2 = V(current_ye2);
Ce2 = C(current_ye2);
Vba = V(current_yba);
Cba = C(current_yba);

% 定义概率函数
sab_forward = @(m1, d1) 1 - normcdf(log(2)*sqrt(m1/Vab).*(Cab - d1./m1));
se1_forward = @(m1, d1) 1 - normcdf(log(2)*sqrt(m1/Ve1).*(Ce1 - d1./m1));
sba_backward = @(m1, d2) 1 - normcdf(log(2)*sqrt((M-m1)/Vba).*(Cba - d2./(M-m1)));
se2_backward = @(m1, d2) 1 - normcdf(log(2)*sqrt((M-m1)/Ve2).*(Ce2 - d2./(M-m1)));

% 目标函数定义
s = @(m1, d1, d2) 1 - (1 - sab_forward(m1, d1)) .* se1_forward(m1, d1) .* ...
    (1 - sba_backward(m1, d2)) .* se2_backward(m1, d2);

% 近似函数
x=@(m1, d1, d2)se1_forward(m1, d1)-sab_forward(m1, d1)*se1_forward(m1, d1);
y=@(m1, d1, d2)se2_forward(m1, d2)-sba_forward(m1, d2)*se2_forward(m1, d2);
p_majorize=@(m1, d1, d2)x*y;

% 设置参数范围
min_m1 = 100;
max_m1 = 900;
min_d1 = 20;
max_d1 = 1000;
min_d2 = 20;
max_d2 = 1000;

% 随机重启机制
num_restarts = 100;  % 重启次数
global_best_m1 = 0;
global_best_d1 = 0;
global_best_d2 = 0;
global_best_s = inf;

fprintf('开始随机重启优化，共%d次重启\n', num_restarts);
M=1000;
for restart = 1:num_restarts
    % 随机生成初始点
    init_m1 = randi([min_m1, max_m1]);
    init_d1 = randi([min_d1, max_d1]);
    init_d2 = randi([min_d2, max_d2]);
    
    fprintf('\n重启 %d: 初始值 m1=%d, d1=%d, d2=%d\n', restart, init_m1, init_d1, init_d2);
    M=1000;
    % 运行坐标下降
    [best_m1, best_d1, best_d2, best_s] = coordinate_descent(s, min_m1, max_m1, min_d1, max_d1, min_d2, max_d2, init_m1, init_d1, init_d2);
    
    fprintf('重启 %d: 最优解 m1=%d, d1=%d, d2=%d, s=%.15f\n', restart, best_m1, best_d1, best_d2, best_s);
    
    % 更新全局最优解
    if best_s < global_best_s
        global_best_s = best_s;
        global_best_m1 = best_m1;
        global_best_d1 = best_d1;
        global_best_d2 = best_d2;
        fprintf('  -> 新的全局最优解找到\n');
    end
end

% ====== 最终结果 ======
fprintf('\n====== 最终结果 ======\n');
fprintf('全局最优 m1 = %d, m2 = %d\n', global_best_m1, M - global_best_m1);
fprintf('全局最优 d1 = %d, d2 = %d\n', global_best_d1, global_best_d2);
fprintf('最小目标函数值 s = %.15f\n', global_best_s);
fprintf('总重启次数: %d\n', num_restarts);

% 坐标下降优化函数（移至文件末尾）
function [best_m1, best_d1, best_d2, best_s] = coordinate_descent(s, min_m1, max_m1, min_d1, max_d1, min_d2, max_d2, init_m1, init_d1, init_d2)
    current_m1 = init_m1;
    current_d1 = init_d1;
    current_d2 = init_d2;
    M=1000;
    current_m2 = M - current_m1;
    current_s = s(current_m1, current_d1, current_d2);
    
    max_iter = 20;
    converged = false;
    
    for iter = 1:max_iter
        prev_s = current_s;
        prev_m1 = current_m1;
        prev_d1 = current_d1;
        prev_d2 = current_d2;
        
        % ====== 优化m1 ======
        best_m1 = current_m1;
        best_s_m1 = current_s;
        
        % 遍历整个m1范围
        for m1 = min_m1:max_m1
            s_val = s(m1, current_d1, current_d2);
            if s_val < best_s_m1
                best_s_m1 = s_val;
                best_m1 = m1;
            end
        end
        
        % 更新m1
        if best_s_m1 < current_s
            current_m1 = best_m1;
            current_m2 = M - current_m1;
            current_s = best_s_m1;
        end
        
        % ====== 优化d1 ======
        best_d1 = current_d1;
        best_s_d1 = current_s;
        
        % 遍历整个d1范围
        for d1 = min_d1:max_d1
            s_val = s(current_m1, d1, current_d2);
            if s_val < best_s_d1
                best_s_d1 = s_val;
                best_d1 = d1;
            end
        end
        
        % 更新d1
        if best_s_d1 < current_s
            current_d1 = best_d1;
            current_s = best_s_d1;
        end
        
        % ====== 优化d2 ======
        best_d2 = current_d2;
        best_s_d2 = current_s;
        
        % 遍历整个d2范围
        for d2 = min_d2:max_d2
            s_val = s(current_m1, current_d1, d2);
            if s_val < best_s_d2
                best_s_d2 = s_val;
                best_d2 = d2;
            end
        end
        
        % 更新d2
        if best_s_d2 < current_s
            current_d2 = best_d2;
            current_s = best_s_d2;
        end
        
        % ====== 收敛判断 ======
        if current_m1 == prev_m1 && current_d1 == prev_d1 && current_d2 == prev_d2
            converged = true;
            break;
        end
    end
    
    best_m1 = current_m1;
    best_d1 = current_d1;
    best_d2 = current_d2;
    best_s = current_s;
end
