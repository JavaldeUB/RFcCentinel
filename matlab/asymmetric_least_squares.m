function baseline = asymmetric_least_squares(y, lambda, p, niter)
% Asymmetric Least Squares baseline correction
% y: vector fila
% lambda: suavizado (p.ej. 1e6)
% p: asimetría (p.ej. 0.001)
% niter: número de iteraciones (p.ej. 10)

    L = length(y);
    D = diff(speye(L), 2);  % Matriz de diferencias de segundo orden
    w = ones(L, 1);         % Pesos iniciales

    for i = 1:niter
        W = spdiags(w, 0, L, L);  % Matriz diagonal de pesos
        Z = W + lambda * (D' * D);
        baseline = Z \ (w .* y(:));  % Resolución del sistema
        w = p * (y(:) > baseline) + (1 - p) * (y(:) < baseline);
    end
end
