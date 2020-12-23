x = [ 0:pi/100:2 * pi ];
y1 = sin(x);
y2 = cos(x);
y3 = tan(x);
figure(1), plot(x, y1, x, y2)
figure(2), plot(x, y3)
figure(3), plot(x, sinh(x))