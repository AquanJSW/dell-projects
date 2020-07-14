I = imread("C:\Temp\withLBL\FRA_600696461EDR_F0731206FHAZ00337M1.png");
FRA_600696461EDR_F0731206FHAZ00337M1;
hs = norm(cross(A, H));
vs = norm(cross(A, V));
hc = dot(A, H);
vc = dot(A, H);
dx = 12; %um
dy = 12;
f = 0.5*(hs*dx + vs*dy) * 0.001; %mm 正确
lens_distortion = zeros(1, 3);
lens_distortion(1) = R(1);
lens_distortion(2) = R(2);
lens_distortion(3) = R(3);
IntrinsicMatrix = [hs,  0,  0
                   0,   vs, 0
                   hc,  vc, 0];
cameraParam = cameraParameters('IntrinsicMatrix', IntrinsicMatrix, ... 
                               'RadialDistortion', [-lens_distortion(1)*10000, -lens_distortion(2), -lens_distortion(3)]);
J = undistortImage(I, cameraParam);
figure; imshowpair(imresize(I,0.5),imresize(J,0.5),'montage');