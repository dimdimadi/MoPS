figure
hold on
in = importdata('opady.prn');
out = importdata('dunaj.prn');
set(gca, 'YScale','log');
plot(in(:,2));
plot(out(:,2),'r');
xlabel('Months');
ylabel('Tritium Concentration (TU)');
legend('opady.prn - measurements in the precipitation.', 'dunaj.prn - measurements in the river.');
hold off