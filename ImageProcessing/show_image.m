img = imread("C:/Users/bolero/git/CV-dc/cv_data/lenna.jpg");
figure(1), imshow(img)
figure(2), imshow(img, 'border', 'tight')



 
x = [ 0:pi/100:10 * pi ];
y = sinc(img);
figure(3), imshow(y)
plot(x, y)