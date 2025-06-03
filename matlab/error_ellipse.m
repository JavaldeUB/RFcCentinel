function error_ellipse(C, mu, varargin)
% ERROR_ELLIPSE - Dibuja una elipse de confianza para una covarianza 2D
%
%   h = error_ellipse(C, mu) dibuja la elipse para la covarianza C (2x2)
%   centrada en mu (1x2). Por defecto es al 95%.
%
%   Opcional: 'conf' para especificar el nivel de confianza (ej. 0.99)
%             'style' para color o estilo de línea

% Parámetros por defecto
conf = 0.95;
style = 'r';

% Procesa argumentos opcionales
for i = 1:2:length(varargin)
    switch lower(varargin{i})
        case 'conf'
            conf = varargin{i+1};
        case 'style'
            style = varargin{i+1};
    end
end

% Verifica tamaños
if ~isequal(size(C), [2 2])
    error('La matriz de covarianza debe ser 2x2.');
end

if length(mu) ~= 2
    error('El centro debe tener dos elementos.');
end

% Eigen descomposición
[vec, val] = eig(C);

% Ángulo de rotación
theta = atan2(vec(2,1), vec(1,1));

% Radio correspondiente al nivel de confianza
chi2_val = chi2inv(conf, 2);
a = sqrt(val(1,1) * chi2_val);
b = sqrt(val(2,2) * chi2_val);

% Parámetro de la elipse
t = linspace(0, 2*pi, 100);
x = a * cos(t);
y = b * sin(t);

% Rotación
R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
xy = R * [x; y];

% Desplazamiento al centro
xy(1,:) = xy(1,:) + mu(1);
xy(2,:) = xy(2,:) + mu(2);

% Dibuja la elipse
plot(xy(1,:), xy(2,:), style, 'LineWidth', 1.5);

% Etiqueta la elipse
%text(mu(1) + a * cos(pi/4), mu(2) + b * sin(pi/4), ...
%     sprintf('%.0f%%', conf*100), ...
%     'FontSize', 11, 'Color', style(1), 'FontWeight', 'bold');

% Asegura que todo lo demás esté en tamaño 18 (puedes llamar esto fuera también)
set(gca, 'FontSize', 18);
xlim([-35 35])
ylim([-20 20])
end
