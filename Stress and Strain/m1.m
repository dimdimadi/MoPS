F = -1000; % Force
L = 1.5; % Beam Length
E = 2.0 * 10^11; % Young's modulus for stainless steel
g = 1; % Beam thickness
d = 0.2; % Beam width

J = (g * d^3) / 12;
h = (F * L^3) / (3 * E * J)