function wrapped = wrapTo2Pi(angle)
% Toolbox-free phase wrapping, including the positive-multiple endpoint.
% Independently implemented for this benchmark, 2026-10-21.
wrapped = mod(angle, 2*pi);
wrapped(wrapped == 0 & angle > 0) = 2*pi;
end
