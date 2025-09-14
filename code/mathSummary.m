function [s, p, msg, details] = mathSummary(x, y)
%MATHSUMMARY Example with multiple outputs of different types.

    arguments
        x (1,1) double
        y (1,1) double
    end

    s = x + y;                            % double
    p = x * y;                            % double
    msg = sprintf('%g + %g = %g', x,y,s); % char (string also fine)
    details = struct('x',x,'y',y,'time',datetime('now')); % struct
end
