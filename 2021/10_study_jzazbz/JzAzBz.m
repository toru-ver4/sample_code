function OUTPUT= JzAzBz(INPUT, mode)
% *** INPUT is the input [X Y Z] or [Jz Az Bz]
% *** INPUT is the input [X Y Z] or [Jz Az Bz]
% *** MOD represents the forward or reverse mode
if mode=='f'
    OUTPUT = XYZ2JzAzBz(INPUT);
elseif mode=='r'
    OUTPUT = JzAzBz2XYZ(INPUT);
else
    disp('Wrong input...');
end

end
function JzAzBz= XYZ2JzAzBz(XYZ_D65)
% *** function JzAzBz = XYZ2JzAzBz(XYZ_D65)
% *** Computes JzAzBz from XYZ_D65
% *** XYZ_D65 is an n by 3 matrix

XYZ_D65(XYZ_D65<0)=0; % XYZ must be non-negative
b=1.15; g=0.66;
XYZ_D65(:,2)= g.*XYZ_D65(:,2)- (g-1)*XYZ_D65(:,1);
XYZ_D65(:,1)= b.*XYZ_D65(:,1)- (b-1)*XYZ_D65(:,3);
  M1 = [0.41478972 0.579999 0.014648;-0.20151 1.120649 0.0531008; -0.0166008 0.2648 0.6684799];
  M2 = [0.5 0.5 0; 3.524000  -4.066708 +0.542708; 0.199076 +1.096799  -1.295875];
n = 2610/2^14; p = (1.7)*2523/2^5;  
c1 = 3424/2^12; c2 = 2413/2^7; c3 = 2392/2^7;
PQ = @(x)((c1+c2*(x./10000).^n) ./ (1+c3*(x./10000).^n)).^p;
XYZ_2_IzAzBz = @(x)PQ(x * M1') * M2';
IzAzBz = XYZ_2_IzAzBz(XYZ_D65);
Iz= IzAzBz(:,1);

d = -0.56;
Jz = (1+d)*Iz./(1+d*Iz)-1.6295499532821566e-11;
JzAzBz =[Jz IzAzBz(:,2) IzAzBz(:,3)];
end

function XYZ_D65 = JzAzBz2XYZ(JzAzBz)
% *** function XYZ_D65 = JzAzBz2XYZ(JzAzBz)
% *** Computes XYZ_D65 from JzAzBz
% *** JzAzBz is an n by 3 matrix

JzAzBz(:,1)=JzAzBz(:,1)+1.6295499532821566e-11;
d = -0.56; 
Iz = JzAzBz(:,1)./(1+d-d*JzAzBz(:,1));
IzAzBz = [Iz JzAzBz(:,2) JzAzBz(:,3)];
M1inv = inv([0.41478972 0.579999 0.014648;-0.20151 1.120649 0.0531008; -0.0166008 0.2648 0.6684799]);
M2inv = inv([0.5 0.5 0; 3.524000  -4.066708 +0.542708; 0.199076 +1.096799  -1.295875]);
n = 2610/2^14; p = (1.7)*2523/2^5;  
c1 = 3424/2^12; c2 = 2413/2^7; c3 = 2392/2^7;
PQinv = @(x)10000*((c1-x.^(1/p))./(c3*x.^(1/p)-c2)).^(1/n);
IzAzBz_2_XYZ = @(x)PQinv(x * M2inv') * M1inv';
XYZ_D65 = IzAzBz_2_XYZ(IzAzBz);
b=1.15; g=0.66;
XYZ_D65(:,1)= (XYZ_D65(:,1)+ (b-1)*XYZ_D65(:,3))./b;
XYZ_D65(:,2)= (XYZ_D65(:,2)+ (g-1)*XYZ_D65(:,1))./g;
end